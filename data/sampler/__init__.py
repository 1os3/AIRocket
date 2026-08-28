"""采样模块：工况参数 LHS/随机采样与确定性种子派生

模块: data/sampler/__init__.py
依赖: data.sampler.sampler
读取配置: 无（见 sampler.py 文件头）
对外接口:
    - SamplePlan: 单样本采样计划
    - plan_samples(cfg) -> list[SamplePlan]
"""

from data.sampler.sampler import SamplePlan, plan_samples

__all__ = ["SamplePlan", "plan_samples"]
