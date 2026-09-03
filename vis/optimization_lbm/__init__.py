"""优化翼型 LBM 复核与对比可视化模块重导出

模块: vis/optimization_lbm/__init__.py
依赖: vis.optimization_lbm.optimization_lbm
读取配置: 无
对外接口:
    - render_optimization_lbm_evaluation(cfg, result_path=None) -> dict
"""

from vis.optimization_lbm.optimization_lbm import render_optimization_lbm_evaluation

__all__ = ["render_optimization_lbm_evaluation"]
