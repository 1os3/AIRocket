from pathlib import Path


def check_inference_request(dataset, indices: list[int] | None,
                            checkpoint: str | Path | None) -> list[int]:
    # 校验对象: render_model_inference 的数据集与 indices —— 至少选择一个存在的样本编号
    assert len(dataset) > 0, "训练缓存无样本，无法执行模型推理可视化"
    selected = dataset.indices if indices is None else indices
    missing = sorted(set(selected) - set(dataset.indices))
    assert selected and not missing, f"推理样本编号为空或不存在：{missing}"
    # 校验对象: render_model_inference 的 checkpoint —— 非空路径必须指向已有文件
    if checkpoint is not None:
        assert Path(checkpoint).is_file(), f"推理 checkpoint 不存在：{checkpoint}"
    return selected


def check_inference_checkpoint(state: dict, expected_fingerprint: str) -> None:
    # 校验对象: 模型推理 checkpoint —— 必须含模型权重且与当前训练缓存严格匹配
    assert isinstance(state, dict) and isinstance(state.get("model"), dict), (
        "推理 checkpoint 缺少 model 权重字典")
    assert state.get("cache_fingerprint") == expected_fingerprint, (
        "推理 checkpoint 与当前训练缓存不匹配")
