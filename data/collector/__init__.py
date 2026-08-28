"""采集模块：端到端数据集生成编排（断点续采）

模块: data/collector/__init__.py
依赖: data.collector.collector
读取配置: 无（见 collector.py 文件头）
对外接口:
    - collect(cfg) -> dict(统计信息)
"""

from data.collector.collector import collect

__all__ = ["collect"]
