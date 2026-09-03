import torch


def check_force_inputs(fields: dict, polygon: torch.Tensor, cfg,
                       u_lb, reynolds: float, surface_offset_cells: float) -> None:
    # 校验对象: compute_force_coefficients 的 fields —— 单样本场量须完整且匹配配置网格
    missing = {"ux", "uy", "p"} - set(fields)
    assert not missing, f"气动力积分缺少场量：{missing}"
    expected = (1, cfg.grid.ny, cfg.grid.nx)
    assert all(tuple(fields[name].shape) == expected for name in ("ux", "uy", "p")), (
        f"气动力积分场量形状必须为 {expected}")
    # 校验对象: compute_force_coefficients 的 polygon —— 连续轮廓至少含三点且坐标有限
    assert polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2, (
        "polygon 必须为至少三点的 (N,2) 连续轮廓")
    tensors = [fields[name] for name in ("ux", "uy", "p")]
    assert all(bool(torch.isfinite(value).all()) for value in (*tensors, polygon)), (
        "气动力积分输入含 NaN/Inf")
    # 校验对象: u_lb/reynolds/surface_offset_cells —— 归一化分母与外侧采样距离须为正
    assert float(torch.as_tensor(u_lb)) > 0.0, "u_lb 必须 > 0"
    assert reynolds > 0.0 and surface_offset_cells > 0.0, (
        "reynolds 与 surface_offset_cells 必须 > 0")
