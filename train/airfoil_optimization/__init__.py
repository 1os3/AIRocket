"""NACA 翼型参数端到端优化模块重导出

模块: train/airfoil_optimization/__init__.py
依赖: train.airfoil_optimization.airfoil_optimization
读取配置: 无
对外接口:
    - optimize_airfoil(cfg, checkpoint=None) -> dict: 用训练模型优化 NACA 参数
"""

from train.airfoil_optimization.airfoil_optimization import optimize_airfoil

__all__ = ["optimize_airfoil"]
