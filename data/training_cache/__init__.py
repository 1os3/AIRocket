"""训练缓存模块重导出

模块: data/training_cache/__init__.py
依赖: data.training_cache.training_cache
读取配置: 无
对外接口:
    - prepare_training_cache(cfg) -> dict
    - TrainingFlowDataset: 只读训练缓存 Dataset
"""

from data.training_cache.training_cache import TrainingFlowDataset, prepare_training_cache

__all__ = ["TrainingFlowDataset", "prepare_training_cache"]
