"""训练包声明

模块: train/__init__.py
依赖: train.engine, train.losses
读取配置: 无
对外接口:
    - train_model / evaluate_model / smoke_test: 训练系统入口
    - compute_flow_losses / reconstruct_fields: 损失与场重建
"""

from train.engine import evaluate_model, smoke_test, train_model
from train.losses import compute_flow_losses, reconstruct_fields

__all__ = ["train_model", "evaluate_model", "smoke_test",
           "compute_flow_losses", "reconstruct_fields"]
