"""流场损失模块重导出

模块: train/losses/__init__.py
依赖: train.losses.losses
读取配置: 无
对外接口:
    - reconstruct_fields(prediction, batch) -> dict
    - compute_flow_losses(prediction, batch, cfg, progress) -> dict
"""

from train.losses.losses import compute_flow_losses, reconstruct_fields

__all__ = ["reconstruct_fields", "compute_flow_losses"]
