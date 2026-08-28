"""翼型模块：NACA 四位数参数化生成与 GPU 栅格化

模块: data/airfoil/__init__.py
依赖: data.airfoil.airfoil
读取配置: 无（见 airfoil.py 文件头）
对外接口:
    - naca4_polygon(m, p, t, n_points, device) -> (K, 2) 张量
    - build_airfoil_geometry(cfg, m, p, t, aoa_deg, device) -> (mask, sdf)
    - build_airfoil_mask(cfg, m, p, t, aoa_deg, device) -> (ny, nx) bool 张量
"""

from data.airfoil.airfoil import build_airfoil_geometry, build_airfoil_mask, naca4_polygon

__all__ = ["naca4_polygon", "build_airfoil_geometry", "build_airfoil_mask"]
