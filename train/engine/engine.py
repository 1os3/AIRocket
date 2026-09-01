"""单卡训练、微调、评估、断点恢复与 CPU 最小过拟合验收

模块: train/engine/engine.py
依赖: torch, matplotlib, data.training_cache, model, train.fine_tuning, train.losses
读取配置: device, training.*, evaluation.*, loss.*, vis.cmap, vis.dpi
对外接口:
    - train_model(cfg, checkpoint=None, pretrained_checkpoint=None) -> dict
    - evaluate_model(cfg, checkpoint=None) -> dict
    - smoke_test(cfg) -> dict
说明: CUDA 仅 Transformer 主干进入 autocast；模型自身固定嵌入与解码器为 FP32。
"""

import csv
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.training_cache import TrainingFlowDataset, prepare_training_cache
from model import FlowResidualTransformer
from train.engine.checks import check_dataset, check_training_initialization
from train.fine_tuning import load_pretrained_weights
from train.losses import compute_flow_losses, reconstruct_fields

__all__ = ["train_model", "evaluate_model", "smoke_test"]


def _device(cfg) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _amp_dtype(cfg, device: torch.device):
    if device.type != "cuda":
        return None
    if cfg.training.amp_dtype == "bfloat16":
        return torch.bfloat16
    if cfg.training.amp_dtype == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(device: torch.device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


def _configure_float32_matmul(cfg, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision(cfg.training.float32_matmul_precision)
    print(f"[train] CUDA FP32 matmul={cfg.training.float32_matmul_precision}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _loader(cfg, split: str, shuffle: bool) -> DataLoader:
    dataset = TrainingFlowDataset(cfg, split)
    check_dataset(dataset, split)
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.training.num_workers > 0)


def _optimizer(model: torch.nn.Module, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim < 2 or "norm" in name or "routing_logits" in name else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.training.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.training.lr, betas=(cfg.training.beta1, cfg.training.beta2))


def _gradient_statistics(model: torch.nn.Module, threshold: float) -> tuple[dict, list[tuple[str, float]]]:
    names, rms_values, missing = [], [], 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing += 1
            continue
        names.append(name)
        gradient = parameter.grad.detach().float()
        rms_values.append(gradient.square().mean().sqrt())
    if not rms_values:
        return {
            "grad_rms_min": math.nan, "grad_rms_median": math.nan,
            "grad_rms_max": math.nan, "grad_small_ratio": 0.0,
            "grad_zero_ratio": 0.0, "grad_nonfinite_count": 0,
            "grad_none_count": missing,
        }, []
    rms = torch.stack(rms_values).cpu()
    finite = torch.isfinite(rms)
    finite_rms = rms[finite]
    small = finite & (rms <= threshold)
    count = rms.numel()
    statistics = {
        "grad_rms_min": float(finite_rms.min()) if finite_rms.numel() else math.nan,
        "grad_rms_median": float(finite_rms.median()) if finite_rms.numel() else math.nan,
        "grad_rms_max": float(finite_rms.max()) if finite_rms.numel() else math.nan,
        "grad_small_ratio": float(small.sum()) / count,
        "grad_zero_ratio": float((finite & (rms == 0)).sum()) / count,
        "grad_nonfinite_count": int((~finite).sum()),
        "grad_none_count": missing,
    }
    smallest = sorted(
        ((name, float(value)) for name, value in zip(names, rms) if bool(torch.isfinite(value))),
        key=lambda item: item[1],
    )[:3]
    return statistics, smallest


def _scheduler(optimizer, cfg, total_steps: int):
    warmup_steps = round(total_steps * cfg.training.warmup_ratio)
    minimum = cfg.training.min_lr / cfg.training.lr

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(ratio, 0.0), 1.0)))
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _checkpoint_payload(model, optimizer, scheduler, scaler, cfg,
                        epoch: int, step: int, best: float, manifest: dict,
                        pretrained: dict | None = None) -> dict:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch, "step": step, "best": best, "config": cfg,
        "cache_fingerprint": manifest["fingerprint"],
        "pretrained": pretrained,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None,
                     expected_fingerprint: str | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if expected_fingerprint is not None:
        assert checkpoint["cache_fingerprint"] == expected_fingerprint, "checkpoint 与训练缓存不匹配"
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _write_log(output: Path, row: dict, stem: str = "metrics") -> None:
    output.mkdir(parents=True, exist_ok=True)
    with open(output / f"{stem}.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output / f"{stem}.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def _evaluate(model, loader, cfg, device: torch.device, amp_dtype) -> tuple[dict, tuple | None]:
    model.eval()
    loss_names = ("data", "edge_data", "total", "gradient", "divergence", "momentum", "boundary")
    metric_names = tuple(f"{kind}_{metric}_{field}" for kind in ("model", "baseline")
                         for metric in ("rmse", "relative_l2") for field in ("ux", "uy", "p"))
    sums = {key: 0.0 for key in (*loss_names, *metric_names)}
    count, visual = 0, None
    for batch in loader:
        batch = _move_batch(batch, device)
        with _autocast(device, amp_dtype):
            prediction = model(batch["inputs"], batch["conditions"])
        losses = compute_flow_losses(prediction, batch, cfg, 1.0)
        fluid = (~batch["mask"]).unsqueeze(1).expand_as(prediction)
        size = prediction.shape[0]
        for name in loss_names:
            sums[name] += float(losses[name]) * size
        for channel, field in enumerate(("ux", "uy", "p")):
            channel_fluid = fluid[:, channel]
            target = batch["target"][:, channel]
            for kind, error in (("model", prediction[:, channel] - target),
                                ("baseline", -target)):
                squared = error.square().masked_fill(~channel_fluid, 0.0).sum()
                points = channel_fluid.sum().clamp_min(1)
                energy = target.square().masked_fill(~channel_fluid, 0.0).sum().clamp_min(1.0e-12)
                sums[f"{kind}_rmse_{field}"] += float((squared / points).sqrt()) * size
                sums[f"{kind}_relative_l2_{field}"] += float((squared / energy).sqrt()) * size
        count += size
        if visual is None:
            visual = (prediction.detach().cpu(), {key: value.detach().cpu() if torch.is_tensor(value) else value
                                                   for key, value in batch.items()})
    return {key: value / max(count, 1) for key, value in sums.items()}, visual


def _render_visual(visual: tuple | None, output: Path, limit: int, cmap: str, dpi: int) -> None:
    if visual is None or limit == 0:
        return
    import matplotlib.pyplot as plt

    prediction, batch = visual
    predicted = reconstruct_fields(prediction, batch)["normalized"]
    target = reconstruct_fields(batch["target"], batch)["normalized"]
    baseline = reconstruct_fields(torch.zeros_like(prediction), batch)["normalized"]
    names = ("ux/u_lb", "uy/u_lb", "p/u_lb²")
    output.mkdir(parents=True, exist_ok=True)
    for sample in range(min(limit, prediction.shape[0])):
        figure, axes = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
        for channel, name in enumerate(names):
            panels = (baseline[sample, channel], predicted[sample, channel],
                      target[sample, channel], (predicted - target)[sample, channel].abs())
            for column, field in enumerate(panels):
                image = axes[channel, column].imshow(field.numpy(), origin="lower", cmap=cmap)
                axes[channel, column].set_title(
                    f"{name} · {('Baseline','Prediction','GT','Absolute error')[column]}")
                axes[channel, column].axis("off")
                figure.colorbar(image, ax=axes[channel, column], fraction=0.046)
        index = int(batch["index"][sample])
        figure.savefig(output / f"sample_{index:08d}.png", dpi=dpi)
        plt.close(figure)


def train_model(cfg, checkpoint: str | None = None,
                pretrained_checkpoint: str | Path | None = None) -> dict:
    """执行正式单卡训练；可选择完整续训或仅加载兼容预训练模型参数。"""
    check_training_initialization(checkpoint, pretrained_checkpoint)
    device = _device(cfg)
    amp_dtype = _amp_dtype(cfg, device)
    _configure_float32_matmul(cfg, device)
    train_loader = _loader(cfg, "train", True)
    try:
        val_loader = _loader(cfg, "val", False)
    except AssertionError:
        val_loader = _loader(cfg, "train", False)
    base_model = FlowResidualTransformer(cfg).to(device)
    pretrained = None
    if pretrained_checkpoint is not None:
        pretrained = load_pretrained_weights(base_model, pretrained_checkpoint)
        print(f"[finetune] loaded={pretrained['loaded_keys']}/{pretrained['total_keys']} "
              f"parameters={pretrained['loaded_ratio']:.2%} "
              f"shape_mismatch={len(pretrained['shape_mismatch_keys'])} "
              f"unexpected={len(pretrained['unexpected_keys'])}")
    model = base_model
    if cfg.training.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"[train] torch.compile 不可用，回退 eager：{exc}")
    optimizer = _optimizer(base_model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation)
    total_steps = cfg.training.max_steps or cfg.training.epochs * steps_per_epoch
    print(f"[train] micro_batch={cfg.training.batch_size} "
          f"gradient_accumulation={cfg.training.gradient_accumulation} "
          f"nominal_effective_batch="
          f"{cfg.training.batch_size * cfg.training.gradient_accumulation} "
          f"optimizer_steps_per_epoch={steps_per_epoch}")
    scheduler = _scheduler(optimizer, cfg, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if device.type == "cuda" else None
    start_epoch, global_step, best = 0, 0, float("inf")
    if checkpoint:
        state = _load_checkpoint(checkpoint, base_model, optimizer, scheduler, scaler,
                                 train_loader.dataset.manifest["fingerprint"])
        start_epoch, global_step, best = state["epoch"] + 1, state["step"], state["best"]
        pretrained = state.get("pretrained")
    output = Path(cfg.training.output_dir)
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            progress = min(global_step / max(total_steps - 1, 1), 1.0)
            with _autocast(device, amp_dtype):
                prediction = model(batch["inputs"], batch["conditions"])
            losses = compute_flow_losses(prediction, batch, cfg, progress)
            scaled_loss = losses["total"] / cfg.training.gradient_accumulation
            if scaler and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            update = (batch_index + 1) % cfg.training.gradient_accumulation == 0 \
                or batch_index + 1 == len(train_loader)
            if not update:
                continue
            if scaler and scaler.is_enabled():
                scaler.unscale_(optimizer)
            log_step = (global_step + 1) % cfg.training.log_every == 0
            gradient_stats, smallest = ({}, [])
            if cfg.training.gradient_monitor and log_step:
                gradient_stats, smallest = _gradient_statistics(
                    base_model, cfg.training.gradient_small_threshold)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip)
            if gradient_stats:
                norm = float(gradient_norm.detach())
                gradient_stats["grad_global_norm"] = norm
                gradient_stats["grad_clip_scale"] = (
                    min(1.0, cfg.training.grad_clip / max(norm, torch.finfo(torch.float32).tiny))
                    if math.isfinite(norm) else math.nan)
            if scaler and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if log_step:
                row = {"epoch": epoch, "step": global_step, "lr": scheduler.get_last_lr()[0],
                       **{key: float(value.detach()) for key, value in losses.items()}}
                _write_log(output, row)
                print(f"[train] epoch={epoch} global_step={global_step}/{total_steps} "
                      f"loss={row['total']:.6g} data={row['data']:.6g}")
                if gradient_stats:
                    _write_log(output, {"epoch": epoch, "step": global_step, **gradient_stats},
                               "gradients")
                    print(f"[grad] norm={gradient_stats['grad_global_norm']:.3e} "
                          f"rms_median={gradient_stats['grad_rms_median']:.3e} "
                          f"small={gradient_stats['grad_small_ratio']:.1%} "
                          f"zero={gradient_stats['grad_zero_ratio']:.1%} "
                          f"nonfinite={gradient_stats['grad_nonfinite_count']} "
                          f"smallest={smallest}")
            if global_step >= total_steps:
                stop = True
                break
        metrics, visual = _evaluate(model, val_loader, cfg, device, amp_dtype)
        _write_log(output, {"epoch": epoch, "step": global_step, **metrics}, "validation")
        payload = _checkpoint_payload(base_model, optimizer, scheduler, scaler, cfg, epoch,
                                      global_step, min(best, metrics["data"]),
                                      train_loader.dataset.manifest, pretrained)
        if (epoch + 1) % cfg.training.checkpoint_every == 0 or stop:
            _atomic_save(payload, output / "latest.pt")
        if metrics["data"] < best:
            best = metrics["data"]
            payload["best"] = best
            _atomic_save(payload, output / "best.pt")
            _render_visual(visual, output / "visuals", cfg.evaluation.num_visualizations,
                           cfg.vis.cmap, cfg.vis.dpi)
        print(f"[eval] epoch={epoch} data={metrics['data']:.6g} "
              f"ux_RMSE={metrics['model_rmse_ux']:.6g} "
              f"baseline={metrics['baseline_rmse_ux']:.6g}")
        if stop:
            break
    return {"step": global_step, "best": best, "output_dir": str(output),
            "pretrained": pretrained}


def evaluate_model(cfg, checkpoint: str | None = None) -> dict:
    """加载 checkpoint，在 test（为空则 val/train）上评估并渲染。"""
    device = _device(cfg)
    amp_dtype = _amp_dtype(cfg, device)
    _configure_float32_matmul(cfg, device)
    loader = None
    for split in ("test", "val", "train"):
        try:
            loader = _loader(cfg, split, False)
            break
        except AssertionError:
            continue
    model = FlowResidualTransformer(cfg).to(device)
    path = Path(checkpoint) if checkpoint else Path(cfg.training.output_dir) / "best.pt"
    _load_checkpoint(path, model, expected_fingerprint=loader.dataset.manifest["fingerprint"])
    metrics, visual = _evaluate(model, loader, cfg, device, amp_dtype)
    _render_visual(visual, Path(cfg.training.output_dir) / "evaluation",
                   cfg.evaluation.num_visualizations, cfg.vis.cmap, cfg.vis.dpi)
    print(f"[evaluate] {metrics}")
    return metrics


def smoke_test(cfg) -> dict:
    """CPU FP32 完整模型单样本前后向、20 步过拟合和 checkpoint 往返。"""
    assert _device(cfg).type == "cpu", "smoke 模式必须使用 CPU"
    prepare_training_cache(cfg)
    dataset = TrainingFlowDataset(cfg, "all")
    check_dataset(dataset, "all")
    batch = _move_batch(next(iter(DataLoader(dataset, batch_size=1, shuffle=False))), torch.device("cpu"))
    model = FlowResidualTransformer(cfg).float()
    routes = 2 * cfg.model.depth + 1
    assert model.routing_logits.numel() == routes * (routes + 1) // 2, "AttnRes 参数数目错误"
    weights = model.routing_weights()
    assert torch.allclose(weights.sum(1), torch.ones(routes), atol=2.0e-7), "AttnRes 行权重和不为 1"
    forbidden = (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm)
    assert not any(isinstance(module, forbidden) for module in model.modules()), "模型含非 RMSNorm"
    optimizer = _optimizer(model, cfg)
    initial, best = None, float("inf")
    steps = cfg.training.max_steps or 20
    completed_steps = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["inputs"], batch["conditions"])
        assert prediction.dtype == torch.float32, "CPU 冒烟输出必须为 FP32"
        losses = compute_flow_losses(prediction, batch, cfg, step / max(steps - 1, 1))
        assert all(torch.isfinite(value) for value in losses.values()), "冒烟损失出现 NaN/Inf"
        losses["total"].backward()
        if step >= 1:
            pixel_grads = [stage.shuffle_conv.weight.grad for stage in model.decoder]
            assert all(grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
                       for grad in pixel_grads), "三级 PixelShuffle 未全部获得有效梯度"
            route_grad = model.routing_logits.grad
            assert route_grad is not None and torch.isfinite(route_grad).all() \
                and bool(route_grad.abs().sum() > 0), "静态 AttnRes 未获得有效梯度"
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        data_value = float(losses["data"].detach())
        initial = data_value if initial is None else initial
        best = min(best, data_value)
        completed_steps = step + 1
        print(f"[smoke] step={step + 1}/{steps} "
              f"total={float(losses['total'].detach()):.6g} data={data_value:.6g}")
        threshold = 1.0 - cfg.training.smoke_min_improvement
        if completed_steps >= 2 and best <= initial * threshold:
            break
    assert best <= initial * threshold, (
        f"单样本过拟合未达到 {cfg.training.smoke_min_improvement:.1%}："
        f"initial={initial} best={best}")
    with torch.no_grad():
        prediction = model(batch["inputs"], batch["conditions"])
    output = Path(cfg.training.output_dir)
    payload = _checkpoint_payload(model, optimizer, None, None, cfg, 0, completed_steps,
                                  best, dataset.manifest)
    _atomic_save(payload, output / "smoke.pt")
    restored = FlowResidualTransformer(cfg).float()
    _load_checkpoint(output / "smoke.pt", restored,
                     expected_fingerprint=dataset.manifest["fingerprint"])
    with torch.no_grad():
        restored_prediction = restored(batch["inputs"], batch["conditions"])
    assert torch.allclose(prediction, restored_prediction, rtol=1.0e-5, atol=1.0e-6), (
        "checkpoint 往返后模型输出不一致")
    visual = (restored_prediction, batch)
    _render_visual(visual, output / "visuals", cfg.evaluation.num_visualizations,
                   cfg.vis.cmap, cfg.vis.dpi)
    result = {"steps": completed_steps, "initial_data": initial, "best_data": best,
              "improvement": 1.0 - best / initial}
    print(f"[smoke] 完成：{result}")
    return result
