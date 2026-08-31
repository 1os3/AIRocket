import torch


def check_model_inputs(fields: torch.Tensor, conditions: torch.Tensor, cfg) -> None:
    # 校验对象: FlowResidualTransformer.forward 入参 fields/conditions —— 形状与网格配置一致
    assert fields.ndim == 4, "fields 必须为 (B,C,H,W)"
    assert fields.shape[1] == cfg.model.input_channels, "fields 输入通道数与 model.input_channels 不符"
    assert tuple(fields.shape[-2:]) == (cfg.grid.ny, cfg.grid.nx), "fields 空间尺寸与 grid 不符"
    assert conditions.shape == (fields.shape[0], 3), "conditions 必须为 (B,3)"
    assert cfg.grid.nx % cfg.model.patch_size == 0 and cfg.grid.ny % cfg.model.patch_size == 0, (
        "grid.nx/ny 必须能被 model.patch_size 整除")
    assert fields.is_floating_point() and conditions.is_floating_point(), "模型输入必须为浮点张量"
