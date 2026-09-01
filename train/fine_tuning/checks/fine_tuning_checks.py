from collections.abc import Mapping
from pathlib import Path


def check_pretrained_path(path: Path) -> None:
    # 校验对象: load_pretrained_weights 的 checkpoint 路径 —— 必须指向已有文件
    assert path.is_file(), f"预训练 checkpoint 不存在：{path}"


def check_pretrained_checkpoint(path: Path, checkpoint) -> None:
    # 校验对象: load_pretrained_weights 的 checkpoint 内容 —— 必须包含权重映射
    assert isinstance(checkpoint, Mapping), f"预训练 checkpoint 必须是映射：{path}"
    weights = checkpoint.get("model", checkpoint)
    assert isinstance(weights, Mapping) and weights, f"预训练 checkpoint 不含模型权重：{path}"


def check_compatible_weights(path: Path, weights: dict) -> None:
    # 校验对象: load_pretrained_weights 的兼容参数 —— 至少加载一个参数才允许启动微调
    assert weights, f"预训练 checkpoint 没有名称和形状均兼容的模型参数：{path}"
