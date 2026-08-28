"""工况参数采样器：拉丁超立方/随机采样 + 确定性种子派生

模块: data/sampler/sampler.py
依赖: torch, data.sampler.checks.sampler_checks
读取配置: seed, sampler.num_samples, sampler.method, sampler.reynolds, sampler.mach,
          sampler.aoa_deg, sampler.naca_m, sampler.naca_p, sampler.naca_t
对外接口:
    - SamplePlan: 单样本采样计划 dataclass
    - plan_samples(cfg) -> list[SamplePlan]
说明:
    - 采样表只由 (seed, num_samples, method, 各区间) 决定：同配置重跑得到逐位相同
      的计划，因此断点续采按 index 跳过即可，天然不复用任何已采样本。
    - 每样本附带派生种子，供下游（如扰动增强）使用并随元数据入库，供冲突校验。
"""

from dataclasses import dataclass

import torch

from data.sampler.checks.sampler_checks import check_plans

__all__ = ["SamplePlan", "plan_samples"]

# 采样维度与其在 cfg.sampler 中的区间键（顺序即 LHS 的维度顺序，属表结构定义非配置）
_DIMS = ["reynolds", "mach", "aoa_deg", "naca_m", "naca_p", "naca_t"]


@dataclass(frozen=True)
class SamplePlan:
    """单样本的完整采样计划：索引 + 派生种子 + 无量纲工况 + 翼型参数。"""

    index: int
    seed: int
    reynolds: float
    mach: float
    aoa_deg: float
    naca_m: float
    naca_p: float
    naca_t: float


def _lhs_unit(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """LHS：每维独立打乱分层，层内均匀抖动，返回 (n, d) ∈ [0,1)。"""
    perms = torch.stack([torch.randperm(n, generator=gen) for _ in range(d)], dim=1)
    jitter = torch.rand(n, d, generator=gen)
    return (perms + jitter) / n


def plan_samples(cfg) -> list:
    """生成全部样本计划（确定性，与运行次数无关）。

    返回:
        长度 num_samples 的 SamplePlan 列表，按 index 升序
    """
    n, method = cfg.sampler.num_samples, cfg.sampler.method
    gen = torch.Generator().manual_seed(cfg.seed)  # CPU 生成器即可：采样表只算一次
    unit = _lhs_unit(n, len(_DIMS), gen) if method == "lhs" else torch.rand(n, len(_DIMS), generator=gen)
    lo = torch.tensor([getattr(cfg.sampler, k)[0] for k in _DIMS])
    hi = torch.tensor([getattr(cfg.sampler, k)[1] for k in _DIMS])
    values = lo + unit * (hi - lo)
    seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=gen).tolist()
    plans = [SamplePlan(index=i, seed=seeds[i], **dict(zip(_DIMS, values[i].tolist())))
             for i in range(n)]
    check_plans(plans, cfg.sampler)
    return plans
