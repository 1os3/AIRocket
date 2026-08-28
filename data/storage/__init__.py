"""存储模块：LMDB 稳态流场写入器与随机读取 Dataset

模块: data/storage/__init__.py
依赖: data.storage.storage
读取配置: 无（见 storage.py 文件头）
对外接口:
    - FlowFieldWriter: existing_indices / existing_seeds / write / write_meta / close
    - FlowFieldDataset: torch Dataset，支持按位随机读取与 get_by_index
"""

from data.storage.storage import FlowFieldDataset, FlowFieldWriter

__all__ = ["FlowFieldWriter", "FlowFieldDataset"]
