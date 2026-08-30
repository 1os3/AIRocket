"""无粘源面元初值模块：为黏性 LBM 提供近似速度场

模块: data/potential_initializer/__init__.py
依赖: data.potential_initializer.potential_initializer
读取配置: 无（见 potential_initializer.py 文件头）
对外接口:
    - build_potential_initial(cfg, plans, u_lb, masks, device) -> dict
"""

from data.potential_initializer.potential_initializer import build_potential_initial

__all__ = ["build_potential_initial"]
