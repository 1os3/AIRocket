"""训练损失、验证损失与梯度健康状态的四面板曲线

模块: vis/training_curves/training_curves.py
依赖: config, matplotlib(Agg), numpy, vis.training_curves.checks
读取配置: training.output_dir, vis.dpi, vis.training_curve_smoothing
对外接口:
    - render_training_curves(cfg) -> Path
    - main() -> None
说明: 中断续训产生重复 step 时保留最后一条；损坏的末行跳过，不影响已完成记录。
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vis.training_curves.checks import check_training_log

__all__ = ["render_training_curves", "main"]


def _read_records(path: Path, required: bool = False) -> list[dict]:
    if required:
        check_training_log(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    records, skipped = {}, 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                records[int(record["step"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1
    if skipped:
        print(f"[curves] 警告：{path.name} 跳过 {skipped} 条损坏记录")
    result = [records[step] for step in sorted(records)]
    assert result, f"{path} 没有含有效 step 的记录"
    return result


def _series(records: list[dict], key: str, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(row["step"]), float(row[key])) for row in records if key in row]
    if not pairs:
        return np.empty(0), np.empty(0)
    values = np.asarray(pairs, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    if positive:
        valid &= values[:, 1] > 0.0
    return values[valid, 0], values[valid, 1]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1 or values.size == 0:
        return values
    finite = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float64)
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="full")[:values.size]
    count = np.convolve(finite.astype(np.float64), kernel, mode="full")[:values.size]
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _plot(ax, records: list[dict], key: str, label: str, window: int,
          positive: bool = False, linestyle: str = "-") -> bool:
    steps, values = _series(records, key, positive)
    if not steps.size:
        return False
    line = ax.plot(steps, values, alpha=0.18, linewidth=0.8, linestyle=linestyle)[0]
    ax.plot(steps, _smooth(values, window), label=label, linewidth=1.8,
            linestyle=linestyle, color=line.get_color())
    return True


def _finish_axis(ax, title: str, ylabel: str, log: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if log:
        ax.set_yscale("log")
    if ax.lines:
        ax.legend(fontsize=8)


def render_training_curves(cfg) -> Path:
    """读取训练日志并生成 loss/梯度诊断图。

    参数:
        cfg: 配置对象，读取 training.output_dir 与 vis 的绘图参数
    返回:
        生成的 training_curves.png 路径
    """
    output = Path(cfg.training.output_dir)
    metrics = _read_records(output / "metrics.jsonl", required=True)
    validation = _read_records(output / "validation.jsonl")
    gradients = _read_records(output / "gradients.jsonl")
    window = cfg.vis.training_curve_smoothing
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for key, label in (("total", "train total"), ("data", "train data")):
        _plot(axes[0, 0], metrics, key, label, window, positive=True)
    for key, label in (("total", "validation total"), ("data", "validation data")):
        _plot(axes[0, 0], validation, key, label, 1, positive=True, linestyle="--")
    _finish_axis(axes[0, 0], "Primary losses", "Loss", log=True)

    for key in ("edge_data", "gradient", "divergence", "momentum", "boundary"):
        _plot(axes[0, 1], metrics, key, key, window, positive=True)
    _finish_axis(axes[0, 1], "Raw loss components", "Loss", log=True)

    for key, label in (("grad_global_norm", "global norm"),
                       ("grad_rms_min", "parameter RMS min"),
                       ("grad_rms_median", "parameter RMS median"),
                       ("grad_rms_max", "parameter RMS max")):
        _plot(axes[1, 0], gradients, key, label, window, positive=True)
    if not gradients:
        axes[1, 0].text(0.5, 0.5, "No gradients.jsonl yet", ha="center", va="center",
                        transform=axes[1, 0].transAxes)
    _finish_axis(axes[1, 0], "Gradient scale before clipping", "Gradient", log=True)

    for records, key, label in ((gradients, "grad_small_ratio", "small gradient ratio"),
                                (gradients, "grad_zero_ratio", "zero gradient ratio"),
                                (gradients, "grad_clip_scale", "clip scale"),
                                (metrics, "physics_scale", "physics warmup")):
        _plot(axes[1, 1], records, key, label, window)
    axes[1, 1].set_ylim(-0.02, 1.02)
    _finish_axis(axes[1, 1], "Gradient health and schedules", "Ratio")

    figure.suptitle(f"Training diagnostics · smoothing window={window}")
    path = output / "training_curves.png"
    figure.savefig(path, dpi=cfg.vis.dpi)
    plt.close(figure)
    print(f"[curves] 已渲染 {path}")
    return path


def main() -> None:
    """加载统一配置并执行训练曲线 CLI。"""
    from config import load_config

    parser = argparse.ArgumentParser(description="训练损失与梯度曲线可视化")
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/train_smoke.yaml）")
    args = parser.parse_args()
    render_training_curves(load_config(env_path=args.env))


if __name__ == "__main__":
    main()
