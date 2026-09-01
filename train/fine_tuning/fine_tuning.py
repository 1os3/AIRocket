"""兼容参数部分加载与独立优化状态的模型微调入口

模块: train/fine_tuning/fine_tuning.py
依赖: torch, train.fine_tuning.checks
读取配置: 无
对外接口:
    - load_pretrained_weights(model, checkpoint) -> dict
    - fine_tune_model(cfg, pretrained_checkpoint) -> dict
说明: 只迁移名称和形状兼容的模型参数；优化器、调度器与训练步数从零开始。
"""

from pathlib import Path

import torch

from train.fine_tuning.checks import (
    check_compatible_weights,
    check_pretrained_checkpoint,
    check_pretrained_path,
)

__all__ = ["load_pretrained_weights", "fine_tune_model"]


def _model_weights(checkpoint: dict) -> dict:
    weights = checkpoint["model"] if isinstance(checkpoint.get("model"), dict) else checkpoint
    return {name.removeprefix("_orig_mod."): value for name, value in weights.items()}


def load_pretrained_weights(model: torch.nn.Module, checkpoint: str | Path) -> dict:
    """加载名称与形状均兼容的预训练模型参数。

    参数:
        model: 当前待微调模型
        checkpoint: 训练 checkpoint 或裸 state_dict 路径
    返回:
        加载数量、覆盖率及跳过参数明细；不恢复任何优化状态
    """
    path = Path(checkpoint)
    check_pretrained_path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    check_pretrained_checkpoint(path, state)
    source = _model_weights(state)
    target = model.state_dict()
    compatible = {
        name: value for name, value in source.items()
        if name in target and torch.is_tensor(value) and value.shape == target[name].shape
    }
    check_compatible_weights(path, compatible)
    shape_mismatch = sorted(
        name for name, value in source.items()
        if name in target and (not torch.is_tensor(value) or value.shape != target[name].shape))
    missing = sorted(set(target) - set(compatible))
    unexpected = sorted(set(source) - set(target))
    model.load_state_dict(compatible, strict=False)
    loaded_numel = sum(value.numel() for value in compatible.values())
    total_numel = sum(value.numel() for value in target.values())
    return {
        "checkpoint": str(path.resolve()),
        "loaded_keys": len(compatible),
        "total_keys": len(target),
        "loaded_numel": loaded_numel,
        "total_numel": total_numel,
        "loaded_ratio": loaded_numel / max(total_numel, 1),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
    }


def fine_tune_model(cfg, pretrained_checkpoint: str | Path) -> dict:
    """用预训练模型的兼容参数初始化，并从零开始微调优化状态。"""
    from train.engine import train_model

    return train_model(cfg, pretrained_checkpoint=pretrained_checkpoint)
