"""在连续翼型轮廓上积分预测或 LBM 流场的升阻系数

模块: data/aerodynamics/aerodynamics.py
依赖: torch, data.aerodynamics.checks
读取配置: grid.nx, grid.ny, grid.chord, grid.x_le, grid.y_center
对外接口:
    - compute_force_coefficients(fields, polygon, cfg, u_lb, reynolds, surface_offset_cells) -> dict
    - compute_optimization_objective(coefficients, objective_cfg) -> tuple
说明: 两个接口同时供代理模型优化与 LBM 复核使用，确保升阻和目标函数口径一致。
"""

import torch
import torch.nn.functional as F

from data.aerodynamics.checks import check_force_inputs

__all__ = ["compute_force_coefficients", "compute_optimization_objective"]


def _derivatives(field: torch.Tensor, chord: float) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.zeros_like(field)
    dx[..., 1:-1] = 0.5 * (field[..., 2:] - field[..., :-2]) * chord
    dy = 0.5 * (torch.roll(field, -1, dims=-2)
                - torch.roll(field, 1, dims=-2)) * chord
    return dx, dy


def _sample_surface(values: torch.Tensor, points: torch.Tensor, cfg) -> torch.Tensor:
    x = (points[:, 0] * cfg.grid.chord + cfg.grid.x_le) * (2.0 / cfg.grid.nx) - 1.0
    y = (points[:, 1] * cfg.grid.chord + cfg.grid.y_center) * (2.0 / cfg.grid.ny) - 1.0
    grid = torch.stack([x, y], dim=1).view(1, -1, 1, 2)
    return F.grid_sample(
        values, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0]


def compute_force_coefficients(fields: dict, polygon: torch.Tensor, cfg, u_lb,
                               reynolds: float, surface_offset_cells: float) -> dict:
    """沿连续翼型表面积分压力和黏性应力，返回来流坐标系下 Cl/Cd。

    参数:
        fields: 单样本原始格子场，ux/uy/p 形状均为 (1,H,W)
        polygon: 弦长归一化的连续翼型轮廓 (N,2)
        cfg: 完整配置对象，读取 cfg.grid.*
        u_lb: 格子来流速度
        reynolds: 实际格子雷诺数
        surface_offset_cells: 沿外法线向流体侧偏移的网格数
    返回:
        包含 lift 与 drag 两个标量张量的字典
    """
    check_force_inputs(fields, polygon, cfg, u_lb, reynolds, surface_offset_cells)
    end = torch.roll(polygon, -1, dims=0)
    segment = end - polygon
    length = torch.linalg.vector_norm(segment, dim=1).clamp_min(torch.finfo(polygon.dtype).eps)
    tangent = segment / length.unsqueeze(1)
    left_normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=1)
    signed_area = 0.5 * (polygon[:, 0] * end[:, 1] - end[:, 0] * polygon[:, 1]).sum()
    normal = left_normal * torch.where(signed_area < 0.0, 1.0, -1.0)
    midpoint = 0.5 * (polygon + end)
    sample_points = midpoint + normal * (surface_offset_cells / float(cfg.grid.chord))

    speed = torch.as_tensor(u_lb, dtype=fields["ux"].dtype, device=fields["ux"].device)
    ux, uy = fields["ux"] / speed, fields["uy"] / speed
    pressure = fields["p"] / speed.square()
    dux_dx, dux_dy = _derivatives(ux, float(cfg.grid.chord))
    duy_dx, duy_dy = _derivatives(uy, float(cfg.grid.chord))
    surface = _sample_surface(torch.stack([
        pressure, dux_dx, dux_dy, duy_dx, duy_dy,
    ], dim=1), sample_points, cfg)
    pressure, dux_dx, dux_dy, duy_dx, duy_dy = surface
    shear = dux_dy + duy_dx
    viscous = torch.stack([
        2.0 * dux_dx * normal[:, 0] + shear * normal[:, 1],
        shear * normal[:, 0] + 2.0 * duy_dy * normal[:, 1],
    ], dim=1)
    traction = -2.0 * pressure.unsqueeze(1) * normal + (2.0 / reynolds) * viscous
    coefficient = (traction * length.unsqueeze(1)).sum(dim=0)
    return {"lift": coefficient[1], "drag": coefficient[0]}


def compute_optimization_objective(coefficients: dict, objective_cfg) -> tuple:
    """按模式构造基础目标，再统一加入正升力违约项。

    返回:
        `(objective, lift_to_drag, lift_violation)`；最后一项是正升力门槛缺口
    """
    lift, drag = coefficients["lift"], coefficients["drag"]
    drag_magnitude = torch.sqrt(drag.square() + objective_cfg.drag_epsilon ** 2)
    if objective_cfg.mode == "maximize_lift":
        base = -lift
    elif objective_cfg.mode == "minimize_drag":
        base = drag_magnitude
    elif objective_cfg.mode == "maximize_lift_to_drag":
        base = -lift / drag_magnitude
    else:
        base = (objective_cfg.lift_weight * (lift - objective_cfg.target_lift).square()
                + objective_cfg.drag_weight * drag_magnitude)
    violation = torch.relu(
        torch.as_tensor(objective_cfg.minimum_lift, dtype=lift.dtype, device=lift.device) - lift)
    objective = base + objective_cfg.lift_constraint_weight * violation
    return objective, lift / drag_magnitude, violation
