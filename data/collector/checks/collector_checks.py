def check_batch_step_accounting(cfg, out: dict, batch_size: int) -> None:
    """校验粗细网格步数记录完整，且启用网格序列后没有批次静默绕过粗网格。"""
    # 校验对象: _solve_batch 输出 —— 每个输入样本必须各有一条粗/细/总步数记录
    assert all(len(out[key]) == batch_size
               for key in ("coarse_steps", "steps", "total_steps")), "批次步数记录数量不完整"
    # 校验对象: _solve_batch.total_steps —— 总步数必须等于对应样本的粗细步数之和
    assert all(total == coarse + fine for total, coarse, fine in zip(
        out["total_steps"], out["coarse_steps"], out["steps"])), "粗细网格步数记账不一致"
    # 校验对象: _solve_batch.grid_sequence_used —— 执行时每个样本步数须为正，跳过时须全为零
    expected = all(step > 0 for step in out["coarse_steps"]) \
        if out["grid_sequence_used"] else all(step == 0 for step in out["coarse_steps"])
    assert expected, "粗网格执行标记与 coarse_steps 不一致"
    # 校验对象: solver.grid_sequence_policy=always —— 禁止任何批次绕过粗网格
    assert not (cfg.solver.grid_sequence and cfg.solver.grid_sequence_policy == "always") \
        or out["grid_sequence_used"], "grid_sequence_policy=always 时批次不得跳过粗网格"
