def check_plans(plans, sampler_cfg) -> None:
    # 校验对象: plan_samples 输出 —— 派生种子全局唯一（断点续采不复用的前提）
    seeds = [p.seed for p in plans]
    assert len(set(seeds)) == len(seeds), "派生种子出现重复，违反可复现性约定"
    # 校验对象: plan_samples 输出 —— 每个采样值须落在配置区间内（防缩放错误）
    for p in plans:
        for k in ("reynolds", "mach", "aoa_deg", "naca_m", "naca_p", "naca_t"):
            lo, hi = getattr(sampler_cfg, k)
            assert lo <= getattr(p, k) <= hi, f"样本 {p.index} 的 {k}={getattr(p, k)} 越界 [{lo}, {hi}]"
