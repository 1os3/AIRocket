"""模型包声明，公开流场残差 Transformer

模块: model/__init__.py
依赖: model.flow_transformer
读取配置: 无
对外接口:
    - FlowResidualTransformer: 稳态流场残差预测网络
"""

from model.flow_transformer import FlowResidualTransformer

__all__ = ["FlowResidualTransformer"]
