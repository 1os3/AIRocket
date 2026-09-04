"""用训练流场模型的梯度直接优化连续 NACA 四位数参数

模块: train/airfoil_optimization/airfoil_optimization.py
依赖: torch, data.aerodynamics, data.airfoil, data.potential_initializer, model, train.losses,
      train.airfoil_optimization.checks
读取配置: seed, device, grid.*, airfoil.*, solver.potential_*, training.amp_dtype,
          training.float32_matmul_precision, training_data.cache_path, model.*,
          optimization.*
对外接口:
    - optimize_airfoil(cfg, checkpoint=None) -> dict
说明: 网格与模型权重始终冻结；仅 naca_m/naca_p/naca_t 中 fixed=false 的参数更新。
      参数以边界内归一化坐标投影更新，梯度贯穿 SDF、面元基线、模型与升阻积分。
"""

import csv
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from data.aerodynamics import compute_force_coefficients, compute_optimization_objective
from data.airfoil import build_airfoil_geometry, build_airfoil_polygon
from data.potential_initializer import build_potential_initial
from model import FlowResidualTransformer
from train.airfoil_optimization.checks import (
    check_optimization_checkpoint,
    check_optimization_gradients,
    check_optimization_request,
    check_optimization_state,
)
from train.losses import reconstruct_fields

__all__ = ["optimize_airfoil"]


class _BoundedNACA(nn.Module):
    """以单位区间坐标表达有界 NACA 参数，并把 fixed 参数留作常量。"""

    def __init__(self, cfg, device: torch.device):
        super().__init__()
        self.specs = cfg.optimization.parameters
        normalized = {}
        for name in ("naca_m", "naca_p", "naca_t"):
            spec = getattr(self.specs, name)
            if spec.fixed:
                continue
            normalized[name] = nn.Parameter(torch.tensor(
                (spec.initial - spec.bounds[0]) / (spec.bounds[1] - spec.bounds[0]),
                dtype=torch.float32, device=device))
        self.normalized = nn.ParameterDict(normalized)

    def values(self, device: torch.device) -> dict[str, torch.Tensor]:
        """返回边界内的三个标量张量；冻结项不进入优化器。"""
        values = {}
        for name in ("naca_m", "naca_p", "naca_t"):
            spec = getattr(self.specs, name)
            if spec.fixed:
                values[name] = torch.tensor(spec.initial, dtype=torch.float32, device=device)
            else:
                lower = torch.tensor(spec.bounds[0], dtype=torch.float32, device=device)
                width = torch.tensor(
                    spec.bounds[1] - spec.bounds[0], dtype=torch.float32, device=device)
                values[name] = lower + width * self.normalized[name]
        return values

    @torch.no_grad()
    def project(self) -> None:
        """把优化后的归一化参数投影回闭区间，边界初值仍保留有效梯度。"""
        for parameter in self.normalized.values():
            parameter.clamp_(0.0, 1.0)


def _device(cfg) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _autocast(cfg, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if cfg.training.amp_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif cfg.training.amp_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_assets(cfg, checkpoint: Path, manifest_path: Path,
                 device: torch.device) -> tuple[nn.Module, dict]:
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check_optimization_checkpoint(state, manifest)
    model = FlowResidualTransformer(cfg)
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), manifest


def _build_batch(cfg, values: dict, manifest: dict,
                 device: torch.device) -> tuple[dict, torch.Tensor]:
    flow = cfg.optimization.flow
    mask, sdf = build_airfoil_geometry(
        cfg, values["naca_m"], values["naca_p"], values["naca_t"],
        flow.aoa_deg, device)
    u_lb = torch.tensor([flow.u_lb], dtype=torch.float32, device=device)
    plan = SimpleNamespace(**values, aoa_deg=flow.aoa_deg)
    potential = build_potential_initial(cfg, [plan], u_lb, mask.unsqueeze(0), device)
    ux0, uy0 = potential["ux"], potential["uy"]
    p0 = 0.5 * (u_lb.view(1, 1, 1).square() - ux0.square() - uy0.square())
    fluid = (~mask).unsqueeze(0)
    p0 = p0 - p0.masked_fill(~fluid, 0.0).sum((1, 2), keepdim=True) \
        / fluid.sum((1, 2), keepdim=True).clamp_min(1)
    inputs = torch.stack([
        (sdf.float() / float(cfg.grid.chord)).clamp(-1.0, 1.0),
        ux0[0] / u_lb[0], uy0[0] / u_lb[0], p0[0] / u_lb[0].square(),
    ]).unsqueeze(0)
    nu = u_lb * float(cfg.grid.chord) / flow.reynolds
    raw_conditions = torch.stack([
        torch.log(torch.tensor(flow.reynolds, dtype=torch.float32, device=device)),
        torch.log(nu[0]), u_lb[0],
    ]).unsqueeze(0)
    condition_mean = torch.tensor(
        manifest["condition_mean"], dtype=torch.float32, device=device)
    condition_std = torch.tensor(
        manifest["condition_std"], dtype=torch.float32, device=device)
    batch = {
        "inputs": inputs,
        "mask": mask.unsqueeze(0),
        "conditions": (raw_conditions - condition_mean) / condition_std,
        "target_scale": torch.tensor(
            manifest["target_rms"], dtype=torch.float32, device=device).unsqueeze(0),
        "u_lb": u_lb,
        "nu": nu,
        "chord": torch.tensor([cfg.grid.chord], dtype=torch.float32, device=device),
    }
    return batch, build_airfoil_polygon(
        cfg, values["naca_m"], values["naca_p"], values["naca_t"],
        flow.aoa_deg, device)


def _write_results(cfg, checkpoint: Path, history: list[dict], best: dict,
                   positive_lift_feasible: bool) -> dict:
    output = Path(cfg.optimization.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "history.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "objective": asdict(cfg.optimization.objective),
        "flow": {
            "u_lb": cfg.optimization.flow.u_lb,
            "reynolds": cfg.optimization.flow.reynolds,
            "aoa_deg": cfg.optimization.flow.aoa_deg,
        },
        "parameters": asdict(cfg.optimization.parameters),
        "surface_offset_cells": cfg.optimization.surface_offset_cells,
        "best": best,
        "positive_lift_feasible": positive_lift_feasible,
        "best_selection": ("lowest_objective_with_positive_lift"
                           if positive_lift_feasible else "highest_lift_infeasible"),
        "iterations": len(history) - 1,
    }
    with open(output / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def optimize_airfoil(cfg, checkpoint: str | Path | None = None) -> dict:
    """冻结训练模型，在配置边界内直接优化 NACA m/p/t 并保存完整轨迹。

    参数:
        cfg: 完整配置对象，工况、边界、固定项和目标均读取 optimization 段
        checkpoint: 可选权重路径；为空时使用 optimization.checkpoint
    返回:
        含固定工况与最优参数/升阻/目标值的字典
    """
    torch.manual_seed(cfg.seed)
    device = _device(cfg)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision(cfg.training.float32_matmul_precision)
    checkpoint_path = Path(checkpoint or cfg.optimization.checkpoint)
    manifest_path = check_optimization_request(cfg, checkpoint_path)
    model, manifest = _load_assets(cfg, checkpoint_path, manifest_path, device)
    parameters = _BoundedNACA(cfg, device)
    optimizer = torch.optim.Adam(
        parameters.parameters(), lr=cfg.optimization.learning_rate,
        betas=(cfg.optimization.beta1, cfg.optimization.beta2),
        eps=cfg.optimization.epsilon)
    history, best_feasible, best_infeasible = [], None, None

    for step in range(cfg.optimization.steps + 1):
        values = parameters.values(device)
        batch, polygon = _build_batch(cfg, values, manifest, device)
        with _autocast(cfg, device):
            prediction = model(batch["inputs"], batch["conditions"])
        fields = reconstruct_fields(prediction, batch)
        coefficients = compute_force_coefficients(
            fields, polygon, cfg, cfg.optimization.flow.u_lb,
            cfg.optimization.flow.reynolds, cfg.optimization.surface_offset_cells)
        loss, ratio, violation = compute_optimization_objective(
            coefficients, cfg.optimization.objective)
        check_optimization_state(loss, coefficients, values)
        row = {
            "step": step,
            "objective": float(loss.detach()),
            "lift": float(coefficients["lift"].detach()),
            "drag": float(coefficients["drag"].detach()),
            "lift_to_drag": float(ratio.detach()),
            "lift_violation": float(violation.detach()),
            "positive_lift_feasible": bool(
                coefficients["lift"].detach() >= cfg.optimization.objective.minimum_lift),
            **{name: float(value.detach()) for name, value in values.items()},
        }
        history.append(row)
        if row["positive_lift_feasible"]:
            if best_feasible is None or row["objective"] < best_feasible["objective"]:
                best_feasible = dict(row)
        elif best_infeasible is None or row["lift"] > best_infeasible["lift"]:
            best_infeasible = dict(row)
        if step % cfg.optimization.log_every == 0 or step == cfg.optimization.steps:
            print(f"[optimize] step={step}/{cfg.optimization.steps} "
                  f"objective={row['objective']:.6g} Cl={row['lift']:.6g} "
                  f"Cd={row['drag']:.6g} L/D={row['lift_to_drag']:.6g} "
                  f"positive_lift={row['positive_lift_feasible']} "
                  f"m={row['naca_m']:.6f} p={row['naca_p']:.6f} t={row['naca_t']:.6f}")
        if step == cfg.optimization.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        check_optimization_gradients(parameters.named_parameters())
        torch.nn.utils.clip_grad_norm_(
            parameters.parameters(), max_norm=cfg.optimization.grad_clip,
            error_if_nonfinite=True)
        optimizer.step()
        parameters.project()

    feasible = best_feasible is not None
    best = best_feasible if feasible else best_infeasible
    result = _write_results(cfg, checkpoint_path, history, best, feasible)
    status = "已找到正升力可行解" if feasible else "未找到正升力可行解，仅保存最高升力候选"
    print(f"[optimize] {status}；结果写入 {Path(cfg.optimization.output_dir)}")
    return result
