"""流场可视化模块：LMDB 稳态样本 → 多面板 PNG

模块: vis/flowfield/__init__.py
依赖: vis.flowfield.flowfield
读取配置: 无（见 flowfield.py 文件头）
对外接口:
    - render_samples(cfg, indices=None) -> list[Path]
"""

from vis.flowfield.flowfield import render_samples

__all__ = ["render_samples"]
