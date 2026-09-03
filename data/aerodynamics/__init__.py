"""翼型气动力系数与优化目标模块重导出

模块: data/aerodynamics/__init__.py
依赖: data.aerodynamics.aerodynamics
读取配置: 无
对外接口:
    - compute_force_coefficients(fields, polygon, cfg, u_lb, reynolds, surface_offset_cells) -> dict
    - compute_optimization_objective(coefficients, objective_cfg) -> tuple
"""

from data.aerodynamics.aerodynamics import (
    compute_force_coefficients,
    compute_optimization_objective,
)

__all__ = ["compute_force_coefficients", "compute_optimization_objective"]
