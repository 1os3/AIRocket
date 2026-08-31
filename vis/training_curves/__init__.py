"""训练曲线可视化模块重导出

模块: vis/training_curves/__init__.py
依赖: vis.training_curves.training_curves
读取配置: 无
对外接口:
    - render_training_curves(cfg) -> Path
"""

from vis.training_curves.training_curves import render_training_curves

__all__ = ["render_training_curves"]
