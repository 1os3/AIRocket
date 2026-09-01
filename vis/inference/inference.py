"""用训练检查点或确定性随机初始化模型渲染流场推理对比图

模块: vis/inference/inference.py
依赖: matplotlib(Agg), torch, data.training_cache, model, train.losses, vis.inference.checks
读取配置: seed, device, training.amp_dtype, training.float32_matmul_precision,
          training_data.cache_path, model.*, grid.*,
          evaluation.num_visualizations, vis.out_dir, vis.cmap, vis.dpi
对外接口:
    - render_model_inference(cfg, checkpoint=None, indices=None) -> list[Path]
说明: checkpoint 为空时按 seed 随机初始化；非空时校验训练缓存指纹后加载权重。
"""

from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from data.training_cache import TrainingFlowDataset
from model import FlowResidualTransformer
from train.losses import reconstruct_fields
from vis.inference.checks import check_inference_checkpoint, check_inference_request

__all__ = ["render_model_inference"]


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


def _configure_float32_matmul(cfg, device: torch.device) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision(cfg.training.float32_matmul_precision)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _source_name(checkpoint: str | Path | None) -> str:
    return "random" if checkpoint is None else Path(checkpoint).stem


def _load_model(cfg, checkpoint: str | Path | None, fingerprint: str,
                device: torch.device) -> FlowResidualTransformer:
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = FlowResidualTransformer(cfg)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check_inference_checkpoint(state, fingerprint)
        model.load_state_dict(state["model"])
    return model.to(device).eval()


def _masked(field: torch.Tensor, fluid: torch.Tensor) -> np.ndarray:
    return field.masked_fill(~fluid, torch.nan).numpy()


def _render_sample(batch: dict, prediction: torch.Tensor, position: int,
                   output: Path, source: str, cfg) -> Path:
    predicted = reconstruct_fields(prediction, batch)["normalized"]
    target = reconstruct_fields(batch["target"], batch)["normalized"]
    baseline = reconstruct_fields(torch.zeros_like(prediction), batch)["normalized"]
    fluid = ~batch["mask"][position]
    names = ("ux/u_lb", "uy/u_lb", "p/u_lb^2")
    columns = ("Baseline", "Prediction", "GT", "Absolute error")
    figure, axes = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for channel, name in enumerate(names):
        comparison = [
            _masked(fields[position, channel], fluid)
            for fields in (baseline, predicted, target)
        ]
        finite = np.concatenate([field[np.isfinite(field)] for field in comparison])
        vmin, vmax = float(finite.min()), float(finite.max())
        error = np.abs(comparison[1] - comparison[2])
        panels = (*comparison, error)
        for column, field in enumerate(panels):
            limits = {} if column == 3 else {"vmin": vmin, "vmax": vmax}
            image = axes[channel, column].imshow(
                field, origin="lower", cmap=cfg.vis.cmap, **limits)
            axes[channel, column].set_title(f"{name} · {columns[column]}")
            axes[channel, column].axis("off")
            figure.colorbar(image, ax=axes[channel, column], fraction=0.046)
    index = int(batch["index"][position])
    figure.suptitle(f"Model inference · sample={index} · weights={source}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"sample_{index:08d}_{source}.png"
    figure.savefig(path, dpi=cfg.vis.dpi)
    plt.close(figure)
    return path


@torch.inference_mode()
def render_model_inference(cfg, checkpoint: str | Path | None = None,
                           indices: list[int] | None = None) -> list[Path]:
    """对训练缓存样本执行模型前向并渲染基线、预测、真值与误差。

    参数:
        cfg: 完整配置对象
        checkpoint: 可选训练检查点；为空时使用由 cfg.seed 决定的随机初始化
        indices: 可选的实际样本编号；为空时取缓存前 evaluation.num_visualizations 个
    返回:
        已生成 PNG 的路径列表
    """
    dataset = TrainingFlowDataset(cfg, "all")
    requested = None if indices is None else list(dict.fromkeys(indices))
    selected = check_inference_request(dataset, requested, checkpoint)
    limit = cfg.evaluation.num_visualizations
    selected = selected if requested is not None else selected[:limit]
    if not selected:
        print("[inference] evaluation.num_visualizations=0，未渲染样本")
        return []
    positions = {index: position for position, index in enumerate(dataset.indices)}
    batch = default_collate([dataset[positions[index]] for index in selected])
    device = _device(cfg)
    _configure_float32_matmul(cfg, device)
    model = _load_model(cfg, checkpoint, dataset.manifest["fingerprint"], device)
    device_batch = _move_batch(batch, device)
    amp_dtype = _amp_dtype(cfg, device)
    context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())
    with context:
        prediction = model(device_batch["inputs"], device_batch["conditions"])
    cpu_batch = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in device_batch.items()
    }
    prediction = prediction.detach().cpu()
    source = _source_name(checkpoint)
    output = Path(cfg.vis.out_dir) / "inference"
    paths = [
        _render_sample(cpu_batch, prediction, position, output, source, cfg)
        for position in range(prediction.shape[0])
    ]
    print(f"[inference] weights={source} device={device} 已渲染 {len(paths)} 张至 {output}")
    return paths
