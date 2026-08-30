"""采集编排：断点续采 + 分批并行求解 + 入库的端到端流程

模块: data/collector/collector.py
依赖: data.airfoil, data.collector.checks, data.lbm_solver, data.sampler, data.storage
读取配置: device, seed, grid.*, solver.batch_size, solver.boundary, solver.grid_sequence,
          solver.grid_sequence_scale, solver.grid_sequence_conv_tol, solver.grid_sequence_policy,
          solver.initializer,
          solver.potential_panels, solver.potential_blend, solver.potential_speed_limit,
          solver.sample_continuation, solver.continuation_bank_size, sampler.tau_min,
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
from data.collector.checks.collector_checks import check_batch_step_accounting
from data.lbm_solver import CS2, LBMSolver
from data.potential_initializer import build_potential_initial
from data.sampler import plan_samples
from data.storage import FlowFieldWriter

__all__ = ["collect"]

_CS = math.sqrt(CS2)


class _ContinuationBank:
    """在 CPU 保留近期稳态场，并为下一批选择有效参数空间中的最近初值。"""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._features = []
        self._fields = []

    def initial(self, features: torch.Tensor) -> dict | None:
        """按欧氏距离为每个目标选最近场；空库返回 None 以触发势流冷启动。"""
        if not self._features:
            return None
        bank = torch.stack(self._features)
        nearest = torch.cdist(features.cpu(), bank).argmin(dim=1).tolist()
        return {key: torch.stack([self._fields[i][key] for i in nearest])
                for key in ("f", "mask", "rho", "ux", "uy")}

    def add(self, features: torch.Tensor, out: dict) -> None:
        """仅收录严格收敛样本，失败场绝不传播给后续工况。"""
        for i, ok in enumerate(out["converged"]):
            if ok:
                self._features.append(features[i].cpu())
                self._fields.append({key: out[key][i]
                                     for key in ("f", "mask", "rho", "ux", "uy")})
        self._features = self._features[-self._capacity:]
        self._fields = self._fields[-self._capacity:]


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


def _plan_features(cfg, plans) -> torch.Tensor:
    """构造真正影响求解的归一化特征，避免被截断前的目标 Re/Ma 误导近邻选择。"""
    lats = [_derive_lattice_params(cfg, plan) for plan in plans]
    values = torch.tensor([
        [lat["u_lb"], lat["tau"], plan.aoa_deg, plan.naca_m, plan.naca_p, plan.naca_t]
        for plan, lat in zip(plans, lats)], dtype=torch.float64)
    lower, upper = values.amin(dim=0), values.amax(dim=0)
    return (values - lower) / (upper - lower).clamp_min(torch.finfo(values.dtype).eps)


def _continuation_order(plans, features: torch.Tensor, batch_size: int) -> list:
    """Morton 局部排序；首批优先高黏度/低攻角样本，为延续库建立可靠锚点。"""
    quantized = (features.clamp(0.0, 1.0) * 1023).round().to(torch.long)
    codes = torch.zeros(len(plans), dtype=torch.long)
    for bit in range(10):
        for dim in range(features.shape[1]):
            codes |= ((quantized[:, dim] >> bit) & 1) << (bit * features.shape[1] + dim)
    locality = torch.argsort(codes, stable=True).tolist()
    easy = sorted(range(len(plans)),
                  key=lambda i: (-features[i, 1].item(), abs(plans[i].aoa_deg), plans[i].naca_m))
    anchors = easy[:min(batch_size, len(plans))]
    anchor_set = set(anchors)
    return anchors + [i for i in locality if i not in anchor_set]


def _resize_macro_initial(initial: dict | None, size: tuple, device,
                          dtype: torch.dtype) -> dict | None:
    """把宏观场缩放到目标网格；分布函数不能直接跨网格插值。"""
    if initial is None or not all(key in initial for key in ("rho", "ux", "uy")):
        return None
    rho = initial["rho"].to(device=device, dtype=dtype)
    if rho.ndim == 3:
        rho = rho.unsqueeze(1)
    resized = {"rho": torch.nn.functional.interpolate(
        rho, size=size, mode="bilinear", align_corners=False)}
    for key in ("ux", "uy"):
        field = initial[key].to(device=device, dtype=dtype)
        resized[key] = torch.nn.functional.interpolate(
            field.unsqueeze(1), size=size, mode="bilinear",
            align_corners=False).squeeze(1)
    return resized


def _uniform_initial(size: tuple, u_lb: torch.Tensor) -> dict:
    """构造均匀来流宏观场，供禁用势流或粗网格失败时逐样本回退。"""
    b = u_lb.shape[0]
    rho = torch.ones(b, 1, *size, dtype=u_lb.dtype, device=u_lb.device)
    ux = u_lb.view(b, 1, 1).expand(b, *size).clone()
    uy = torch.zeros_like(ux)
    return {"rho": rho, "ux": ux, "uy": uy}


def _mask_initial_velocity(initial: dict, masks: torch.Tensor) -> dict:
    """插值会把固体内外速度混合；目标网格固体节点必须重新置零。"""
    initial["ux"] = initial["ux"].masked_fill(masks, 0.0)
    initial["uy"] = initial["uy"].masked_fill(masks, 0.0)
    return initial


def _run_batch(cfg, solver, writer, plans, device, initial=None,
               return_state: bool = False, progress_offset: int = 0,
               progress_total: int | None = None) -> tuple:
    """跑一批样本并逐样本入库，返回本批统计。"""
    sequence = None
    geometry_cfg = cfg
    use_grid_sequence = cfg.solver.grid_sequence and (
        initial is None or cfg.solver.grid_sequence_policy == "always")
    if use_grid_sequence:
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
    coarse_steps = [0] * len(plans)
    if sequence is not None:
        # 跨样本延续库保存的是细网格状态：先缩到粗网格，不允许它绕过网格序列。
        coarse_initial = _resize_macro_initial(
            initial, (sequence.grid.ny, sequence.grid.nx), device, dtype)
        if coarse_initial is None and cfg.solver.initializer == "potential":
            coarse_initial = build_potential_initial(
                sequence, plans, u_lb, masks, device)
        if coarse_initial is not None:
            coarse_initial = _mask_initial_velocity(coarse_initial, masks)
        coarse_solver = LBMSolver(sequence, device)
        coarse = coarse_solver.run_batch(masks, u_lb, tau, sdfs, initial=coarse_initial)
        coarse_steps = coarse["steps"]
        geoms = [build_airfoil_geometry(cfg, p.naca_m, p.naca_p, p.naca_t, p.aoa_deg, device)
                 for p in plans]
        masks = torch.stack([g[0] for g in geoms])
        sdfs = torch.stack([g[1] for g in geoms]) if cfg.solver.boundary == "bouzidi" else None
        fine_size = (cfg.grid.ny, cfg.grid.nx)
        fine_from_coarse = _resize_macro_initial(coarse, fine_size, device, dtype)
        if all(coarse["converged"]):
            initial = fine_from_coarse
        else:
            # 只让粗网格失败的样本回退；已收敛的粗解仍继续用于同批其他样本。
            fallback = _resize_macro_initial(initial, fine_size, device, dtype)
            if fallback is None and cfg.solver.initializer == "potential":
                fallback = build_potential_initial(cfg, plans, u_lb, masks, device)
            if fallback is None:
                fallback = _uniform_initial(fine_size, u_lb)
            coarse_ok = torch.tensor(coarse["converged"], device=device).view(-1, 1, 1)
            initial = {
                "rho": torch.where(coarse_ok.unsqueeze(1), fine_from_coarse["rho"],
                                   fallback["rho"]),
                "ux": torch.where(coarse_ok, fine_from_coarse["ux"], fallback["ux"]),
                "uy": torch.where(coarse_ok, fine_from_coarse["uy"], fallback["uy"]),
            }
        initial = _mask_initial_velocity(initial, masks)
    elif initial is None and cfg.solver.initializer == "potential":
        initial = build_potential_initial(cfg, plans, u_lb, masks, device)
    out = solver.run_batch(masks, u_lb, tau, sdfs, initial=initial, return_state=return_state)
    out["coarse_steps"] = coarse_steps
    out["total_steps"] = [coarse_step + fine_step
                          for coarse_step, fine_step in zip(coarse_steps, out["steps"])]
    out["grid_sequence_used"] = sequence is not None
    check_batch_step_accounting(cfg, out, len(plans))
    stats = {"written": 0, "failed": 0}
    for j, (p, lat) in enumerate(zip(plans, lats)):
        progress = progress_offset + j + 1
        progress_text = f"进度 {progress}/{progress_total}，" if progress_total is not None else ""
        route = f"粗 {out['coarse_steps'][j]} + 细 {out['steps'][j]}" \
            if out["grid_sequence_used"] else f"粗跳过（稳态延续初值）+ 细 {out['steps'][j]}"
        if not out["converged"][j]:
            print(f"[collector] {progress_text}样本ID {p.index} 未收敛/发散"
                  f"（{route} 步），丢弃")
            stats["failed"] += 1
            continue
        fields = {k: out[k][j] for k in ("rho", "ux", "uy", "p", "mask")}
        writer.write(p, fields, {**lat, "steps": out["total_steps"][j],
                                 "coarse_steps": out["coarse_steps"][j],
                                 "fine_steps": out["steps"][j],
                                 "converged": True, "chord": cfg.grid.chord})
        print(f"[collector] {progress_text}样本ID {p.index} 收敛于 {out['total_steps'][j]} 步"
              f"（{route}），已写入")
        stats["written"] += 1
    return stats, out


def collect(cfg) -> dict:
    """端到端采集主流程。同配置重跑自动续采；最终返回统计 dict。"""
    device = _resolve_device(cfg.device)
    writer = FlowFieldWriter(cfg)
    plans = plan_samples(cfg)
    done = writer.existing_indices()
    todo = [p for p in plans if p.index not in done]
    all_features = _plan_features(cfg, plans) if cfg.solver.sample_continuation else None
    if all_features is not None and todo:
        todo_features = all_features[torch.tensor([p.index for p in todo])]
        order = _continuation_order(todo, todo_features, cfg.solver.batch_size)
        todo = [todo[i] for i in order]
    pending_seeds = {p.seed for p in todo}
    # 校验对象: 续采种子安全 —— 待采样本的派生种子不得与库内已有种子重复
    assert not (pending_seeds & writer.existing_seeds()), (
        "待采样本种子与库内重复：配置（seed/num_samples/method/区间）已被改动，请用新库")
    print(f"[collector] device={device} 计划 {len(plans)} 样本，已完成 {len(done)}，待采 {len(todo)}")
    if all_features is not None and todo:
        print("[collector] 已按工况邻近性重排待采顺序；样本ID仍是采样表中的稳定编号，"
              "因此日志中的ID不会连续")
    if cfg.solver.grid_sequence:
        policy = "有稳态延续时跳过粗网格" \
            if cfg.solver.grid_sequence_policy == "auto" else "每批强制执行粗网格"
        print(f"[collector] 粗细网格策略={cfg.solver.grid_sequence_policy}（{policy}）")
    writer.write_meta()
    solver = LBMSolver(cfg, device)
    stats = {"written": 0, "skipped_done": len(done), "failed": 0}
    bs = cfg.solver.batch_size
    bank = _ContinuationBank(cfg.solver.continuation_bank_size) \
        if cfg.solver.sample_continuation else None
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        features = all_features[torch.tensor([p.index for p in batch])] \
            if all_features is not None else None
        initial = bank.initial(features) if bank is not None else None
        batch_stats, out = _run_batch(
            cfg, solver, writer, batch, device, initial, return_state=bank is not None,
            progress_offset=i, progress_total=len(todo))
        if bank is not None:
            bank.add(features, out)
        stats["written"] += batch_stats["written"]
        stats["failed"] += batch_stats["failed"]
    writer.close()
    stats["total"] = len(plans)
    print(f"[collector] 完成：{stats}")
    return stats
