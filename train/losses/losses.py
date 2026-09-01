"""监督、翼面邻域侧重、稳态 Navier–Stokes 与精确边界物理损失

模块: train/losses/losses.py
依赖: torch, train.losses.checks
读取配置: loss.*
对外接口:
    - reconstruct_fields(prediction, batch) -> dict
    - compute_flow_losses(prediction, batch, cfg, progress) -> dict
说明: 所有场、差分和归约均强制 FP32；x 为非周期方向，y 与 LBM 一致按周期处理。
"""

import torch
import torch.nn.functional as F

from train.losses.checks import check_loss_inputs

__all__ = ["reconstruct_fields", "compute_flow_losses"]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    return value.masked_fill(~expanded, 0.0).sum() / expanded.sum().clamp_min(1)


def _dx(field: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(field)
    result[..., 1:-1] = 0.5 * (field[..., 2:] - field[..., :-2])
    return result


def _dy(field: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2))


def _laplacian(field: torch.Tensor) -> torch.Tensor:
    result = torch.roll(field, -1, dims=-2) + torch.roll(field, 1, dims=-2) - 2.0 * field
    result[..., 1:-1] += field[..., 2:] + field[..., :-2] - 2.0 * field[..., 1:-1]
    return result


def _valid_stencil(fluid: torch.Tensor) -> torch.Tensor:
    valid = fluid & torch.roll(fluid, 1, dims=-2) & torch.roll(fluid, -1, dims=-2)
    valid[..., 1:-1] &= fluid[..., :-2] & fluid[..., 2:]
    valid[..., 0] = False
    valid[..., -1] = False
    return valid


def reconstruct_fields(prediction: torch.Tensor, batch: dict) -> dict:
    """把尺度化残差还原为全场，压力按流体域零均值固定规约。"""
    residual = prediction.float() * batch["target_scale"].float().unsqueeze(-1).unsqueeze(-1)
    inputs = batch["inputs"].float()
    normalized = torch.stack([
        inputs[:, 1] + residual[:, 0],
        inputs[:, 2] + residual[:, 1],
        inputs[:, 3] + residual[:, 2],
    ], dim=1)
    fluid = ~batch["mask"]
    pressure = normalized[:, 2]
    pressure_mean = pressure.masked_fill(~fluid, 0.0).sum((1, 2)) / fluid.sum((1, 2)).clamp_min(1)
    normalized[:, 2] = pressure - pressure_mean[:, None, None]
    normalized[:, 0] = normalized[:, 0].masked_fill(~fluid, 0.0)
    normalized[:, 1] = normalized[:, 1].masked_fill(~fluid, 0.0)
    u_lb = batch["u_lb"].float().view(-1, 1, 1)
    return {
        "residual": residual,
        "normalized": normalized,
        "ux": normalized[:, 0] * u_lb,
        "uy": normalized[:, 1] * u_lb,
        "p": normalized[:, 2] * u_lb.square(),
        "fluid": fluid,
    }


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor, delta: float) -> torch.Tensor:
    differences = [
        F.huber_loss(_dx(prediction), _dx(target), reduction="none", delta=delta),
        F.huber_loss(_dy(prediction), _dy(target), reduction="none", delta=delta),
    ]
    return sum(_masked_mean(item, valid.unsqueeze(1)) for item in differences)


def _physics_losses(fields: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    ux, uy, pressure = fields["ux"], fields["uy"], fields["p"]
    valid = _valid_stencil(fields["fluid"])
    chord = batch["chord"].float().view(-1, 1, 1)
    u_lb = batch["u_lb"].float().view(-1, 1, 1)
    nu = batch["nu"].float().view(-1, 1, 1)
    dux_dx, dux_dy = _dx(ux), _dy(ux)
    duy_dx, duy_dy = _dx(uy), _dy(uy)
    divergence = (dux_dx + duy_dy) * chord / u_lb.clamp_min(1.0e-8)
    scale = chord / u_lb.square().clamp_min(1.0e-8)
    momentum_x = (ux * dux_dx + uy * dux_dy + _dx(pressure)
                  - nu * _laplacian(ux)) * scale
    momentum_y = (ux * duy_dx + uy * duy_dy + _dy(pressure)
                  - nu * _laplacian(uy)) * scale
    divergence_loss = _masked_mean(divergence.square(), valid)
    momentum = torch.stack([momentum_x, momentum_y], dim=1)
    momentum_loss = _masked_mean(momentum.square(), valid.unsqueeze(1))
    return divergence_loss, momentum_loss


def _boundary_loss(fields: dict, batch: dict) -> torch.Tensor:
    normalized = fields["normalized"]
    inlet = (normalized[:, 0, :, 0] - 1.0).square().mean() \
        + normalized[:, 1, :, 0].square().mean()
    outlet = (normalized[:, :, :, -1] - normalized[:, :, :, -2]).square().mean()
    periodic = (normalized[:, :, 0, :] - normalized[:, :, -1, :]).square().mean()
    velocity = normalized[:, :2]
    grid = batch["boundary"].float().unsqueeze(1)
    wall_velocity = F.grid_sample(
        velocity, grid, mode="bilinear", padding_mode="border", align_corners=False)
    wall = wall_velocity.square().mean()
    return inlet + outlet + periodic + wall


def compute_flow_losses(prediction: torch.Tensor, batch: dict, cfg,
                        progress: float = 1.0) -> dict:
    """计算总损失及各分量；progress 为 [0,1] 训练进度，用于物理项升权。"""
    check_loss_inputs(prediction, batch)
    prediction, target = prediction.float(), batch["target"].float()
    fluid = (~batch["mask"]).unsqueeze(1)
    point_data = F.huber_loss(
        prediction, target, reduction="none", delta=cfg.loss.huber_delta)
    data = _masked_mean(point_data, fluid)
    distance_cells = batch["inputs"][:, :1].float().abs() \
        * batch["chord"].float().view(-1, 1, 1, 1)
    edge = fluid & (distance_cells <= cfg.loss.edge_band_cells)
    edge_data = _masked_mean(point_data, edge)
    valid = _valid_stencil(~batch["mask"])
    gradient = _gradient_loss(prediction, target, valid, cfg.loss.huber_delta)
    fields = reconstruct_fields(prediction, batch)
    divergence, momentum = _physics_losses(fields, batch)
    boundary = _boundary_loss(fields, batch)
    warmup = cfg.loss.physics_warmup_ratio
    physics_scale = 1.0 if warmup == 0.0 else min(1.0, progress / warmup)
    total = (cfg.loss.data_weight * data + cfg.loss.edge_data_weight * edge_data
             + cfg.loss.gradient_weight * gradient + physics_scale * (
        cfg.loss.divergence_weight * divergence + cfg.loss.momentum_weight * momentum
        + cfg.loss.boundary_weight * boundary))
    return {"total": total, "data": data, "edge_data": edge_data, "gradient": gradient,
            "divergence": divergence, "momentum": momentum, "boundary": boundary,
            "physics_scale": torch.tensor(physics_scale, device=prediction.device)}
