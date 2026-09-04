from pathlib import Path

import torch


def check_optimization_request(cfg, checkpoint: str | Path) -> Path:
    # 校验对象: optimize_airfoil 的 checkpoint 与训练缓存 —— 两项产物都必须存在
    checkpoint_path = Path(checkpoint)
    manifest_path = Path(cfg.training_data.cache_path) / "manifest.json"
    assert checkpoint_path.is_file(), f"优化 checkpoint 不存在：{checkpoint_path}"
    assert manifest_path.is_file(), f"训练缓存 manifest 不存在：{manifest_path}"
    return manifest_path


def check_optimization_checkpoint(state: dict, manifest: dict) -> None:
    # 校验对象: optimize_airfoil 加载的 checkpoint —— 权重、缓存指纹与统计量必须完整匹配
    assert isinstance(state, dict) and isinstance(state.get("model"), dict), (
        "优化 checkpoint 缺少 model 权重字典")
    assert state.get("cache_fingerprint") == manifest.get("fingerprint"), (
        "优化 checkpoint 与当前 training_data 缓存指纹不匹配")
    for key in ("condition_mean", "condition_std", "target_rms"):
        values = manifest.get(key)
        assert isinstance(values, list) and len(values) == 3, f"训练缓存缺少三通道 {key}"


def check_optimization_state(loss: torch.Tensor, coefficients: dict,
                             parameters: dict) -> None:
    # 校验对象: 每步目标、升阻系数与 NACA 参数 —— 非有限值会使后续更新失去意义
    values = torch.stack([loss, *coefficients.values(), *parameters.values()])
    assert bool(torch.isfinite(values).all()), "翼型优化出现 NaN/Inf，请检查工况与参数范围"


def check_optimization_gradients(named_parameters) -> None:
    # 校验对象: loss.backward 后的待优化参数梯度 —— 坏梯度不得进入裁剪器和 Adam 状态
    gradients = [(name, parameter.grad) for name, parameter in named_parameters]
    missing = [name for name, gradient in gradients if gradient is None]
    invalid = [name for name, gradient in gradients
               if gradient is not None and not bool(torch.isfinite(gradient).all())]
    assert not missing, f"翼型优化参数缺少梯度：{missing}"
    assert not invalid, f"翼型优化参数梯度出现 NaN/Inf：{invalid}"
