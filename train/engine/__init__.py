"""训练引擎模块重导出

模块: train/engine/__init__.py
依赖: train.engine.engine
读取配置: 无
对外接口:
    - train_model(cfg, checkpoint=None) -> dict
    - evaluate_model(cfg, checkpoint=None) -> dict
    - smoke_test(cfg) -> dict
"""

from train.engine.engine import evaluate_model, smoke_test, train_model

__all__ = ["train_model", "evaluate_model", "smoke_test"]
