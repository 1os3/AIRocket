"""模型推理可视化模块重导出

模块: vis/inference/__init__.py
依赖: vis.inference.inference
读取配置: 无
对外接口:
    - render_model_inference(cfg, checkpoint=None, indices=None) -> list[Path]
"""

from vis.inference.inference import render_model_inference

__all__ = ["render_model_inference"]
