"""训练曲线可视化模块重导出

模块: vis/training_curves/__init__.py
依赖: vis.training_curves.training_curves
读取配置: 无
对外接口:
    - render_training_curves(cfg) -> Path
    - main() -> None
"""

__all__ = ["render_training_curves", "main"]


def render_training_curves(cfg):
    """惰性导入实现并渲染训练曲线，避免 `python -m` 重复加载目标模块。"""
    from vis.training_curves.training_curves import render_training_curves as render

    return render(cfg)


def main() -> None:
    """惰性导入训练曲线 CLI，保持包公开接口与实现模块一致。"""
    from vis.training_curves.training_curves import main as run

    run()
