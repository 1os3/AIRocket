"""LBM 求解模块：D2Q9 MRT 全 GPU 批量稳态求解器

模块: data/lbm_solver/__init__.py
依赖: data.lbm_solver.lbm_solver
读取配置: 无（见 lbm_solver.py 文件头）
对外接口:
    - LBMSolver: run_batch(masks, u_lb, tau) -> dict(rho, ux, uy, p, steps, converged)
    - CS2: 格子声速平方常量
"""

from data.lbm_solver.lbm_solver import CS2, LBMSolver

__all__ = ["LBMSolver", "CS2"]
