"""用同工况 LBM 复核优化翼型并渲染初始与最优流场对比

模块: vis/optimization_lbm/optimization_lbm.py
依赖: matplotlib(Agg), numpy, torch, data.aerodynamics, data.airfoil,
      data.collector, data.lbm_solver, data.sampler, vis.optimization_lbm.checks
读取配置: seed, device, grid.*, airfoil.*, sampler.tau_min, solver.*,
          optimization.output_dir, optimization.surface_offset_cells,
          optimization.objective.*,
          vis.out_dir, vis.cmap, vis.dpi, vis.fields
对外接口:
    - render_optimization_lbm_evaluation(cfg, result_path=None) -> dict
说明: 初始与优化翼型在同一批次、同一工况和同一求解设置下复核；不写入原始 LMDB。
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.aerodynamics import compute_force_coefficients, compute_optimization_objective
from data.airfoil import build_airfoil_polygon
from data.collector import solve_plans
from data.lbm_solver import CS2
from data.sampler import SamplePlan
from vis.optimization_lbm.checks import (
    check_lbm_evaluation_path,
    check_lbm_evaluation_result,
    check_lbm_solver_output,
)

__all__ = ["render_optimization_lbm_evaluation"]


def _case_parameters(result: dict) -> list[dict]:
    names = ("naca_m", "naca_p", "naca_t")
    initial = {name: float(result["parameters"][name]["initial"]) for name in names}
    optimized = {name: float(result["best"][name]) for name in names}
    return [initial, optimized]


def _plans(result: dict, cfg) -> tuple:
    flow = result["flow"]
    mach = flow["u_lb"] / math.sqrt(CS2)
    plans = tuple(SamplePlan(
        index=index, seed=cfg.seed + index, reynolds=flow["reynolds"], mach=mach,
        aoa_deg=flow["aoa_deg"], **parameters)
        for index, parameters in enumerate(_case_parameters(result)))
    evaluation_cfg = replace(
        cfg, sampler=replace(cfg.sampler, u_lb_fixed=flow["u_lb"]))
    return plans, evaluation_cfg


def _objective_values(result: dict, cfg) -> dict:
    # 旧版结果没有正升力字段；当前配置只负责补缺，不覆盖结果中已有的目标定义。
    return {**asdict(cfg.optimization.objective), **result["objective"]}


def _case_metrics(out: dict, index: int, plan, lattice: dict,
                  result: dict, cfg) -> dict:
    metrics = {
        "converged": bool(out["converged"][index]),
        "coarse_steps": int(out["coarse_steps"][index]),
        "fine_steps": int(out["steps"][index]),
        "total_steps": int(out["total_steps"][index]),
        "parameters": {name: float(getattr(plan, name))
                       for name in ("naca_m", "naca_p", "naca_t")},
        "lattice": {name: float(lattice[name]) for name in
                    ("u_lb", "tau", "nu", "reynolds_lattice", "mach_lattice")},
    }
    if not metrics["converged"]:
        return metrics
    fields = {name: out[name][index].unsqueeze(0) for name in ("ux", "uy", "p")}
    polygon = build_airfoil_polygon(
        cfg, plan.naca_m, plan.naca_p, plan.naca_t, plan.aoa_deg, "cpu")
    surface_offset = result.get(
        "surface_offset_cells", cfg.optimization.surface_offset_cells)
    coefficients = compute_force_coefficients(
        fields, polygon, cfg, lattice["u_lb"], lattice["reynolds_lattice"],
        surface_offset)
    objective_cfg = SimpleNamespace(**_objective_values(result, cfg))
    objective, ratio, violation = compute_optimization_objective(
        coefficients, objective_cfg)
    metrics.update({
        "lift": float(coefficients["lift"]),
        "drag": float(coefficients["drag"]),
        "lift_to_drag": float(ratio),
        "objective": float(objective),
        "lift_violation": float(violation),
        "positive_lift_feasible": bool(
            coefficients["lift"] >= objective_cfg.minimum_lift),
    })
    return metrics


def _panel(out: dict, index: int, name: str) -> np.ndarray:
    value = (out["ux"][index].square() + out["uy"][index].square()).sqrt() \
        if name == "speed" else out[name][index]
    array = value.numpy().astype(np.float64)
    return np.where(out["mask"][index].numpy(), np.nan, array)


def _limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    return (lower, upper) if lower < upper else (lower - 1.0, upper + 1.0)


def _metric_title(label: str, metrics: dict) -> str:
    if not metrics["converged"]:
        return f"{label} · 未收敛 · steps={metrics['total_steps']}"
    feasible = "Cl feasible" if metrics["positive_lift_feasible"] else "Cl infeasible"
    return (f"{label} · {feasible} · Cl={metrics['lift']:.5f} · Cd={metrics['drag']:.5f} · "
            f"L/D={metrics['lift_to_drag']:.5f} · steps={metrics['total_steps']}")


def _render(out: dict, metrics: dict, cfg, output: Path) -> Path:
    fields = cfg.vis.fields
    labels = ("Initial", "Optimized")
    arrays = [[_panel(out, row, name) for name in fields] for row in range(2)]
    limits = [_limits([arrays[0][column], arrays[1][column]])
              for column in range(len(fields))]
    figure, axes = plt.subplots(
        2, len(fields), figsize=(4.2 * len(fields), 7.4), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(fields))
    for row in range(2):
        for column, name in enumerate(fields):
            image = axes[row, column].imshow(
                arrays[row][column], origin="lower", cmap=cfg.vis.cmap,
                vmin=limits[column][0], vmax=limits[column][1])
            axes[row, column].set_title(name if row == 0 else "")
            axes[row, column].set_ylabel(labels[row] if column == 0 else "")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.82)
    figure.suptitle("LBM optimization validation\n"
                    + "\n".join(_metric_title(
                        label, metrics[label.lower()]) for label in labels))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "lbm_comparison.png"
    figure.savefig(path, dpi=cfg.vis.dpi)
    plt.close(figure)
    return path


def _save_fields(out: dict, output: Path) -> Path:
    path = output / "lbm_fields.npz"
    np.savez_compressed(path, **{
        f"{label}_{name}": out[name][index].numpy()
        for index, label in enumerate(("initial", "optimized"))
        for name in ("rho", "ux", "uy", "p", "mask")
    })
    return path


def render_optimization_lbm_evaluation(cfg, result_path=None) -> dict:
    """读取优化结果，以相同工况 LBM 求解初始/最优翼型并输出对比产物。

    参数:
        cfg: 完整配置对象
        result_path: 优化 result.json；为空时取 optimization.output_dir/result.json
    返回:
        包含初始/最优 LBM 指标、收敛状态和是否改善的报告字典
    """
    path = Path(result_path or Path(cfg.optimization.output_dir) / "result.json")
    check_lbm_evaluation_path(path)
    with open(path, encoding="utf-8") as file:
        result = json.load(file)
    check_lbm_evaluation_result(result, cfg)
    plans, evaluation_cfg = _plans(result, cfg)
    out, lattice = solve_plans(evaluation_cfg, plans)
    check_lbm_solver_output(out, len(plans), cfg)
    case_metrics = {
        label: _case_metrics(out, index, plans[index], lattice[index], result, cfg)
        for index, label in enumerate(("initial", "optimized"))
    }
    both_converged = all(metrics["converged"] for metrics in case_metrics.values())
    positive_lift = (both_converged
                     and case_metrics["optimized"]["positive_lift_feasible"])
    improved = (positive_lift
                and case_metrics["optimized"]["objective"]
                < case_metrics["initial"]["objective"])
    report = {
        "source_result": str(path.resolve()),
        "objective": _objective_values(result, cfg),
        "surface_offset_cells": result.get(
            "surface_offset_cells", cfg.optimization.surface_offset_cells),
        **case_metrics,
        "both_converged": both_converged,
        "lbm_positive_lift_feasible": positive_lift,
        "lbm_objective_improved": improved,
        "objective_delta": (case_metrics["optimized"]["objective"]
                            - case_metrics["initial"]["objective"])
        if both_converged else None,
    }
    output = Path(cfg.vis.out_dir) / "optimization_lbm"
    image = _render(out, case_metrics, cfg, output)
    fields = _save_fields(out, output)
    with open(output / "lbm_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    if not both_converged:
        verdict = "LBM 未全部收敛，结果无效"
    elif not positive_lift:
        verdict = "LBM 优化翼型未达到正升力门槛，结果无效"
    elif improved:
        verdict = "LBM 正升力有效且目标改善"
    else:
        verdict = "LBM 正升力有效，但未确认目标改善"
    print(f"[optimization-lbm] {verdict}；收敛={both_converged}；"
          f"报告={output / 'lbm_report.json'}；图={image}；场={fields}")
    return report
