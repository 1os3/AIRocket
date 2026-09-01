"""预训练模型微调模块重导出

模块: train/fine_tuning/__init__.py
依赖: train.fine_tuning.fine_tuning
读取配置: 无
对外接口:
    - load_pretrained_weights(model, checkpoint) -> dict
    - fine_tune_model(cfg, pretrained_checkpoint) -> dict
"""

from train.fine_tuning.fine_tuning import fine_tune_model, load_pretrained_weights

__all__ = ["load_pretrained_weights", "fine_tune_model"]
