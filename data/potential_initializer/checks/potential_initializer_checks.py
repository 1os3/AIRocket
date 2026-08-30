import torch


def check_potential_inputs(cfg, plans, u_lb, masks) -> None:
    # 校验对象: build_potential_initial 入参 plans/u_lb/masks —— 批维与网格必须一致
    batch = len(plans)
    assert u_lb.shape == (batch,), "u_lb 须与 plans 等长"
    assert masks.shape == (batch, cfg.grid.ny, cfg.grid.nx), "masks 形状与批次/网格不符"
    assert masks.dtype == torch.bool, "masks 必须是 bool 张量"
