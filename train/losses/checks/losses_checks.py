import torch


def check_loss_inputs(prediction: torch.Tensor, batch: dict) -> None:
    # 校验对象: compute_flow_losses 的 prediction/batch —— 三场、mask 与尺度形状须匹配
    assert prediction.ndim == 4 and prediction.shape[1] == 3, "prediction 必须为 (B,3,H,W)"
    assert prediction.dtype == torch.float32, "模型输出与物理损失必须使用 FP32"
    assert batch["target"].shape == prediction.shape, "target 与 prediction 形状不一致"
    assert batch["mask"].shape == prediction.shape[:1] + prediction.shape[2:], "mask 形状不一致"
    assert batch["target_scale"].shape == (prediction.shape[0], 3), "target_scale 必须为 (B,3)"
