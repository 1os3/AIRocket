"""流场残差 Transformer 模块重导出

模块: model/flow_transformer/__init__.py
依赖: model.flow_transformer.flow_transformer
读取配置: 无
对外接口:
    - FlowResidualTransformer: 12 层静态注意力残差网络
    - RMSNorm / RMSNorm2d: 一维与二维均方根归一化
"""

from model.flow_transformer.flow_transformer import FlowResidualTransformer, RMSNorm, RMSNorm2d

__all__ = ["FlowResidualTransformer", "RMSNorm", "RMSNorm2d"]
