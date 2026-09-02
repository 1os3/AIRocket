"""训练包声明

模块: train/__init__.py
依赖: train.airfoil_optimization, train.engine, train.losses
读取配置: 无
对外接口:
    - train_model / evaluate_model / smoke_test: 训练系统入口
    - compute_flow_losses / reconstruct_fields: 损失与场重建
    - optimize_airfoil: 用训练模型端到端优化 NACA 参数
"""

from train.airfoil_optimization import optimize_airfoil
from train.engine import evaluate_model, smoke_test, train_model
from train.losses import compute_flow_losses, reconstruct_fields

__all__ = ["train_model", "evaluate_model", "smoke_test",
           "compute_flow_losses", "reconstruct_fields", "optimize_airfoil"]
