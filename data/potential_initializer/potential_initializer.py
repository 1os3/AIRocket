"""无粘源面元近似：为黏性 LBM 构造满足翼面不穿透的初始速度场

模块: data/potential_initializer/potential_initializer.py
依赖: torch, data.airfoil, data.potential_initializer.checks
读取配置: grid.nx, grid.ny, grid.chord, grid.x_le, grid.y_center,
          airfoil.n_points, solver.potential_panels, solver.potential_blend,
          solver.potential_speed_limit
对外接口:
    - build_potential_initial(cfg, plans, u_lb, masks, device) -> dict
说明:
    - 使用常强度点源面元近似求解无粘、不可压、无旋流；面元强度由翼面不穿透条件确定。
    - 它不含环量、黏性边界层和尾迹，只作为 LBM 冷启动初值；最终标签仍完全由 MRT-LBM 求得。
    - 初值与均匀来流混合并限速，避免尾缘附近的无粘奇异速度破坏 LBM 稳定性。
"""

import math

import torch

from data.airfoil import build_airfoil_polygon
from data.potential_initializer.checks.potential_initializer_checks import check_potential_inputs

__all__ = ["build_potential_initial"]


def _resample_closed(poly: torch.Tensor, count: int) -> torch.Tensor:
    """按闭合轮廓弧长重采样，避免 NACA 余弦点使面元矩阵过度聚集。"""
    closed = torch.cat([poly, poly[:1]], dim=0)
    delta = closed[1:] - closed[:-1]
    lengths = torch.linalg.vector_norm(delta, dim=1).clamp_min(torch.finfo(poly.dtype).eps)
    cumulative = torch.cat([torch.zeros(1, dtype=poly.dtype, device=poly.device), lengths.cumsum(0)])
    targets = torch.arange(count, dtype=poly.dtype, device=poly.device) * cumulative[-1] / count
    indices = torch.searchsorted(cumulative[1:], targets, right=True).clamp_max(poly.shape[0] - 1)
    fraction = ((targets - cumulative[indices]) / lengths[indices]).unsqueeze(1)
    return closed[indices] + fraction * delta[indices]


def _source_panel_field(cfg, plan, u: torch.Tensor, mask: torch.Tensor) -> tuple:
    """求一个样本的源面元速度场；源面元只负责提供不穿透的粗略全局结构。"""
    dtype, device = u.dtype, u.device
    poly = build_airfoil_polygon(
        cfg, plan.naca_m, plan.naca_p, plan.naca_t, plan.aoa_deg, device).to(dtype)
    poly = _resample_closed(poly, cfg.solver.potential_panels)
    poly = torch.stack([
        poly[:, 0] * cfg.grid.chord + cfg.grid.x_le,
        poly[:, 1] * cfg.grid.chord + cfg.grid.y_center,
    ], dim=1)
    end = torch.roll(poly, -1, dims=0)
    segment = end - poly
    length = torch.linalg.vector_norm(segment, dim=1).clamp_min(torch.finfo(dtype).eps)
    tangent = segment / length.unsqueeze(1)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=1)
    center = 0.5 * (poly + end)

    offset = center[:, None, :] - center[None, :, :]
    radius2 = (offset * offset).sum(dim=2).clamp_min(torch.finfo(dtype).eps)
    velocity = offset * (length / (2.0 * math.pi * radius2)).unsqueeze(2)
    influence = (velocity * normal[:, None, :]).sum(dim=2)
    influence.fill_diagonal_(0.5)
    rhs = -(normal[:, 0] * u)
    regularization = torch.finfo(dtype).eps * influence.shape[0]
    strength = torch.linalg.solve(
        influence + regularization * torch.eye(influence.shape[0], dtype=dtype, device=device), rhs)

    xs = torch.arange(cfg.grid.nx, dtype=dtype, device=device) + 0.5
    ys = torch.arange(cfg.grid.ny, dtype=dtype, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    offset = points[:, None, :] - center[None, :, :]
    core2 = (0.25 * length).square()
    radius2 = (offset * offset).sum(dim=2) + core2.unsqueeze(0)
    induced = (offset * (length * strength / (2.0 * math.pi * radius2)).unsqueeze(2)).sum(dim=1)
    uniform = torch.stack([torch.full_like(points[:, 0], u), torch.zeros_like(points[:, 0])], dim=1)
    field = uniform + cfg.solver.potential_blend * induced
    speed = torch.linalg.vector_norm(field, dim=1).clamp_min(torch.finfo(dtype).eps)
    limit = cfg.solver.potential_speed_limit * u
    field = field * (limit / speed).clamp_max(1.0).unsqueeze(1)
    field = field.reshape(cfg.grid.ny, cfg.grid.nx, 2)
    field[mask] = 0.0
    field[:, 0, 0], field[:, 0, 1] = u, 0.0
    return field[..., 0], field[..., 1]


def build_potential_initial(cfg, plans, u_lb: torch.Tensor,
                            masks: torch.Tensor, device) -> dict:
    """为一批翼型生成无粘源面元初值，返回 rho/ux/uy 三个设备张量。"""
    check_potential_inputs(cfg, plans, u_lb, masks)
    fields = [_source_panel_field(cfg, plan, u_lb[i], masks[i]) for i, plan in enumerate(plans)]
    ux = torch.stack([field[0] for field in fields])
    uy = torch.stack([field[1] for field in fields])
    rho = torch.ones(len(plans), 1, cfg.grid.ny, cfg.grid.nx,
                     dtype=u_lb.dtype, device=device)
    return {"rho": rho, "ux": ux, "uy": uy}
