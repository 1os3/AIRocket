"""采集编排：断点续采 + 分批并行求解 + 入库的端到端流程

模块: data/collector/collector.py
依赖: data.airfoil, data.lbm_solver, data.sampler, data.storage
读取配置: device, seed, grid.*, solver.batch_size, solver.boundary, solver.grid_sequence,
          solver.grid_sequence_scale, solver.grid_sequence_conv_tol, sampler.tau_min,
          sampler.u_lb_max, sampler.u_lb_fixed, sampler.num_samples
对外接口:
    - collect(cfg) -> dict（written / skipped_done / failed / total 统计）
说明:
    - 续采：按 LMDB 已有 index 跳过；采样计划只由 (seed, num_samples, method, 区间) 决定，
      故重算计划必与已入库样本一致，不会复用种子（另以库内种子集合兜底校验）。
    - 并行：待采样本按 solver.batch_size 分批，同批共享一次 GPU 迭代循环；
      批内各样本收敛/发散独立判定，已收敛样本冻结不再更新。
    - 未收敛/发散样本直接丢弃（数据集只保留稳态场）。
"""

import math
from dataclasses import replace

import torch

from data.airfoil import build_airfoil_geometry
from data.lbm_solver import CS2, LBMSolver
from data.sampler import plan_samples
from data.storage import FlowFieldWriter

__all__ = ["collect"]

_CS = math.sqrt(CS2)


def _resolve_device(name: str) -> torch.device:
    """auto 优先 CUDA；本地无 GPU 时回退 CPU（仅开发调试用）。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _derive_lattice_params(cfg, plan) -> dict:
    """把物理工况 (Re, Ma) 折算成格子参数 (U, nu, tau)。

    tau 低于 tau_min 时先抬 U 保 Re（Ma 误差可接受），仍超限再截断 U 保 tau；
    实际生效值以 *_lattice 键返回，随样本写入元数据。
    """
    chord = float(cfg.grid.chord)
    u = cfg.sampler.u_lb_fixed if cfg.sampler.u_lb_fixed is not None \
        else min(plan.mach * _CS, cfg.sampler.u_lb_max)
    tau = 0.5 + u * chord / plan.reynolds / CS2
    if tau < cfg.sampler.tau_min:
        tau = cfg.sampler.tau_min
        if cfg.sampler.u_lb_fixed is None:
            # 非固定速度时可抬 U 保 Re；固定速度时保 U，Re 以 *_lattice 记录实际值
            u = min((tau - 0.5) * CS2 * plan.reynolds / chord, cfg.sampler.u_lb_max)
    nu = (tau - 0.5) * CS2
    return {"u_lb": u, "tau": tau, "nu": nu,
            "reynolds_lattice": u * chord / nu, "mach_lattice": u / _CS}


def _run_batch(cfg, solver, writer, plans, device) -> dict:
    """跑一批样本并逐样本入库，返回本批统计。"""
    sequence = None
    geometry_cfg = cfg
    if cfg.solver.grid_sequence:
        k = cfg.solver.grid_sequence_scale
        g = cfg.grid
        coarse_grid = replace(g, nx=g.nx // k, ny=g.ny // k, chord=max(2, g.chord // k),
                               x_le=max(1, g.x_le // k), y_center=max(1, g.y_center // k))
        coarse_solver_cfg = replace(cfg.solver, grid_sequence=False,
                                    conv_tol=cfg.solver.grid_sequence_conv_tol)
        sequence = replace(cfg, grid=coarse_grid, solver=coarse_solver_cfg)
        geometry_cfg = sequence
    geoms = [build_airfoil_geometry(geometry_cfg, p.naca_m, p.naca_p, p.naca_t, p.aoa_deg, device)
             for p in plans]
    masks = torch.stack([g[0] for g in geoms])
    sdfs = torch.stack([g[1] for g in geoms]) if cfg.solver.boundary == "bouzidi" else None
    lats = [_derive_lattice_params(cfg, p) for p in plans]
    dtype = torch.float64 if cfg.solver.float64 else torch.float32
    u_lb = torch.tensor([l["u_lb"] for l in lats], dtype=dtype, device=device)
    tau = torch.tensor([l["tau"] for l in lats], dtype=dtype, device=device)
    initial = None
    if sequence is not None:
        coarse_solver = LBMSolver(sequence, device)
        coarse = coarse_solver.run_batch(masks, u_lb, tau, sdfs)
        if all(coarse["converged"]):
            fine_size = (cfg.grid.ny, cfg.grid.nx)
            initial = {k: torch.nn.functional.interpolate(coarse[k].unsqueeze(1).to(device),
                                                            size=fine_size, mode="bilinear",
                                                            align_corners=False).squeeze(1)
                       for k in ("rho", "ux", "uy")}
    if sequence is not None:
        geoms = [build_airfoil_geometry(cfg, p.naca_m, p.naca_p, p.naca_t, p.aoa_deg, device)
                 for p in plans]
        masks = torch.stack([g[0] for g in geoms])
        sdfs = torch.stack([g[1] for g in geoms]) if cfg.solver.boundary == "bouzidi" else None
    out = solver.run_batch(masks, u_lb, tau, sdfs, initial=initial)
    stats = {"written": 0, "failed": 0}
    for j, (p, lat) in enumerate(zip(plans, lats)):
        if not out["converged"][j]:
            print(f"[collector] 样本 {p.index} 未收敛/发散（{out['steps'][j]} 步），丢弃")
            stats["failed"] += 1
            continue
        fields = {k: out[k][j] for k in ("rho", "ux", "uy", "p", "mask")}
        writer.write(p, fields, {**lat, "steps": out["steps"][j],
                                 "converged": True, "chord": cfg.grid.chord})
        print(f"[collector] 样本 {p.index} 收敛于 {out['steps'][j]} 步，已写入")
        stats["written"] += 1
    return stats


def collect(cfg) -> dict:
    """端到端采集主流程。同配置重跑自动续采；最终返回统计 dict。"""
    device = _resolve_device(cfg.device)
    writer = FlowFieldWriter(cfg)
    plans = plan_samples(cfg)
    done = writer.existing_indices()
    todo = [p for p in plans if p.index not in done]
    pending_seeds = {p.seed for p in todo}
    # 校验对象: 续采种子安全 —— 待采样本的派生种子不得与库内已有种子重复
    assert not (pending_seeds & writer.existing_seeds()), (
        "待采样本种子与库内重复：配置（seed/num_samples/method/区间）已被改动，请用新库")
    print(f"[collector] device={device} 计划 {len(plans)} 样本，已完成 {len(done)}，待采 {len(todo)}")
    writer.write_meta()
    solver = LBMSolver(cfg, device)
    stats = {"written": 0, "skipped_done": len(done), "failed": 0}
    bs = cfg.solver.batch_size
    for i in range(0, len(todo), bs):
        batch_stats = _run_batch(cfg, solver, writer, todo[i:i + bs], device)
        stats["written"] += batch_stats["written"]
        stats["failed"] += batch_stats["failed"]
    writer.close()
    stats["total"] = len(plans)
    print(f"[collector] 完成：{stats}")
    return stats
