import torch


def check_dataset(dataset, split: str) -> None:
    """校验训练或评估使用的数据集划分。"""
    # 校验对象: train/evaluate 使用的数据集 —— 选定划分必须包含至少一个样本
    assert len(dataset) > 0, f"训练缓存的 {split} 划分为空"


def check_training_initialization(checkpoint, pretrained_checkpoint) -> None:
    """校验正式续训与微调初始化选项的互斥关系。"""
    # 校验对象: train_model 的初始化来源 —— 断点恢复与预训练初始化语义互斥
    assert checkpoint is None or pretrained_checkpoint is None, (
        "checkpoint 断点恢复与 pretrained_checkpoint 微调初始化不能同时使用")


def check_checkpoint_fingerprint(checkpoint: dict, expected: str | None) -> None:
    """校验 checkpoint 与训练缓存的数据指纹。"""
    if expected is None:
        return
    # 校验对象: _load_checkpoint 的 checkpoint —— 模型必须匹配当前训练缓存
    assert checkpoint["cache_fingerprint"] == expected, "checkpoint 与训练缓存不匹配"


def check_smoke_device(device: torch.device) -> None:
    """校验 CPU 冒烟测试的运行设备。"""
    # 校验对象: smoke_test 的运行设备 —— CPU 冒烟必须排除 CUDA/AMP 差异
    assert device.type == "cpu", "smoke 模式必须使用 CPU"


def check_smoke_model(model, cfg) -> None:
    """校验 CPU 冒烟模型的路由、融合初始化与归一化结构。"""
    routes = 2 * cfg.model.depth + 1
    # 校验对象: 主干 routing_logits —— 压缩下三角须包含全部因果路由行
    assert model.routing_logits.numel() == routes * (routes + 1) // 2, (
        "AttnRes 参数数目错误")
    weights = model.routing_weights()
    # 校验对象: 主干 routing_weights —— 每个有效路由行必须为凸组合
    assert torch.allclose(weights.sum(1), torch.ones(routes), atol=2.0e-7), (
        "AttnRes 行权重和不为 1")
    if cfg.model.multiscale_bypass.enabled:
        # 校验对象: bypass_routing_logits —— 旁路须拥有独立的完整最终路由行
        assert model.bypass_routing_logits.numel() == routes, (
            "近壁旁路最终路由行参数数目错误")
        assert model.bypass_routing_logits.data_ptr() != model.routing_logits.data_ptr(), (
            "直接解码与近壁旁路错误共享了最终路由参数")
        output_weights = model.bypass_routing_weights()
        # 校验对象: bypass_routing_weights —— 旁路最终行必须为凸组合
        assert output_weights.shape == (routes,) and torch.allclose(
            output_weights.sum(), output_weights.new_tensor(1.0), atol=2.0e-7), (
            "近壁旁路最终路由行权重和不为 1")
        bypass_routes = cfg.model.multiscale_bypass.depth + 1
        bypass_weights = [branch.routing_weights() for branch in model.multiscale_branches]
        # 校验对象: 多分辨率分支 routing_weights —— 每行必须为凸组合
        assert all(weight.shape == (bypass_routes, bypass_routes)
                   and torch.allclose(weight.sum(1), torch.ones(bypass_routes), atol=2.0e-7)
                   for weight in bypass_weights), "多分辨率旁路 AttnRes 行权重和不为 1"
        # 校验对象: 多分辨率融合 gate —— 初始融合比例必须严格为 0.5/0.5
        assert all(bool(torch.equal(fusion.gate.weight, torch.zeros_like(fusion.gate.weight)))
                   and bool(torch.equal(fusion.gate.bias, torch.zeros_like(fusion.gate.bias)))
                   for fusion in model.multiscale_fusions), "多分辨率融合门控未从 0.5/0.5 初始化"
    forbidden = (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm)
    # 校验对象: 冒烟模型归一化层 —— 模型只允许 RMSNorm
    assert not any(isinstance(module, forbidden) for module in model.modules()), (
        "模型含非 RMSNorm")


def check_smoke_forward(prediction: torch.Tensor, losses: dict) -> None:
    """校验 CPU 冒烟前向的输出精度与损失有限性。"""
    # 校验对象: smoke_test 的 prediction —— CPU 前向必须保持 FP32
    assert prediction.dtype == torch.float32, "CPU 冒烟输出必须为 FP32"
    # 校验对象: smoke_test 的 losses —— 所有损失分量必须为有限数
    assert all(torch.isfinite(value) for value in losses.values()), "冒烟损失出现 NaN/Inf"


def check_smoke_gradients(model, cfg) -> None:
    """校验零初始化输出头完成首步更新后的关键梯度路径。"""
    pixel_grads = [stage.shuffle_conv.weight.grad for stage in model.decoder]
    # 校验对象: 三级 PixelShuffle 的 shuffle_conv —— 每级必须获得有限非零梯度
    assert all(grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
               for grad in pixel_grads), "三级 PixelShuffle 未全部获得有效梯度"
    route_grad = model.routing_logits.grad
    # 校验对象: 主干 routing_logits —— 静态深度路由必须获得有限非零梯度
    assert route_grad is not None and torch.isfinite(route_grad).all() \
        and bool(route_grad.abs().sum() > 0), "静态 AttnRes 未获得有效梯度"
    if not cfg.model.multiscale_bypass.enabled:
        return
    routes = 2 * cfg.model.depth + 1
    direct_output_grad = route_grad[-routes:]
    bypass_output_grad = model.bypass_routing_logits.grad
    # 校验对象: 直接解码最终路由行 —— 全场上采样路径必须获得非零梯度
    assert bool(direct_output_grad.abs().sum() > 0), "直接解码最终路由行未获得有效梯度"
    # 校验对象: 旁路独立最终路由行 —— 近壁上下文路径必须获得有限非零梯度
    assert bypass_output_grad is not None and torch.isfinite(bypass_output_grad).all() \
        and bool(bypass_output_grad.abs().sum() > 0), "近壁旁路独立最终路由行未获得有效梯度"
    branch_grads = [branch.blocks[0].conv1.weight.grad for branch in model.multiscale_branches]
    bypass_route_grads = [branch.routing_logits.grad for branch in model.multiscale_branches]
    gate_grads = [fusion.gate.weight.grad for fusion in model.multiscale_fusions]
    # 校验对象: 多分辨率分支首层卷积 —— 两条局部分支都必须获得有效梯度
    assert all(grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
               for grad in branch_grads), "多分辨率旁路未获得有效梯度"
    # 校验对象: 多分辨率分支 routing_logits —— 两条局部深度路由都必须获得有效梯度
    assert all(grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
               for grad in bypass_route_grads), "多分辨率旁路 AttnRes 未获得有效梯度"
    # 校验对象: 多分辨率融合 gate —— 两个分辨率融合点都必须获得有效梯度
    assert all(grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
               for grad in gate_grads), "多分辨率融合门控未获得有效梯度"


def check_smoke_improvement(initial: float, best: float, minimum: float) -> None:
    """校验单样本过拟合的改善比例。"""
    # 校验对象: smoke_test 的单样本过拟合结果 —— 改善比例必须达到配置阈值
    assert best <= initial * (1.0 - minimum), (
        f"单样本过拟合未达到 {minimum:.1%}：initial={initial} best={best}")


def check_restored_prediction(prediction: torch.Tensor,
                              restored_prediction: torch.Tensor) -> None:
    """校验 checkpoint 往返前后的模型预测。"""
    # 校验对象: smoke_test 的 checkpoint 往返预测 —— 保存和恢复不得改变输出
    assert torch.allclose(prediction, restored_prediction, rtol=1.0e-5, atol=1.0e-6), (
        "checkpoint 往返后模型输出不一致")
