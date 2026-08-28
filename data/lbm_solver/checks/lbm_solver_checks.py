import torch


def check_run_inputs(masks, u_lb, tau, grid_cfg, sdfs=None, boundary="bounce_back") -> None:
    # 校验对象: run_batch 入参 masks —— 须为 (B, ny, nx) bool 张量
    assert masks.ndim == 3 and masks.dtype == torch.bool, "masks 须为 (B, ny, nx) bool"
    assert tuple(masks.shape[1:]) == (grid_cfg.ny, grid_cfg.nx), "masks 形状与 grid 配置不符"
    # 校验对象: run_batch 入参 u_lb / tau —— 须为等长 (B,) 且物理合法
    b = masks.shape[0]
    assert u_lb.shape == (b,) and tau.shape == (b,), "u_lb/tau 须与 masks 批维一致"
    assert (u_lb > 0).all() and (u_lb < 0.5).all(), "u_lb 须在 (0, 0.5)（低马赫假设）"
    assert (tau > 0.5).all(), "tau 必须 > 0.5（否则粘性为负，数值必发散）"
    # 校验对象: run_batch 入参 sdfs —— bouzidi 模式必传，且符号约定与 masks 一致
    if boundary == "bouzidi":
        assert sdfs is not None, "boundary=bouzidi 必须提供 sdfs（有符号距离场）"
        assert sdfs.shape == masks.shape, "sdfs 形状须与 masks 一致"
        assert (sdfs[masks] <= 0).all() and (sdfs[~masks] >= 0).all(), (
            "sdfs 符号约定错误：固体内须 <= 0、流体内须 >= 0")
