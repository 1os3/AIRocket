"""流场可视化：从 LMDB 读取稳态样本并渲染多面板 PNG

模块: vis/flowfield/flowfield.py
依赖: matplotlib(Agg), numpy, data.storage, vis.flowfield.checks.flowfield_checks
读取配置: storage.path, vis.out_dir, vis.cmap, vis.dpi, vis.num_samples, vis.fields
对外接口:
    - render_samples(cfg, indices=None) -> list[Path]（渲染文件路径列表）
说明:
    - 固体节点渲染为 NaN（色图 bad 色），翼型轮廓直观可见。
    - y 轴取格子坐标（向上为正），与求解器 mask/场量布局一致。
    - 全部输出写入 vis.out_dir（工作目录内），不触碰库文件。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器/CI）也能渲染
import matplotlib.pyplot as plt
import numpy as np

from data.storage import FlowFieldDataset
from vis.flowfield.checks.flowfield_checks import check_render_inputs

__all__ = ["render_samples"]


def _panel_array(sample: dict, name: str) -> np.ndarray:
    """按面板名取 (H,W) 数组；speed 由 ux/uy 合成；固体置 NaN。"""
    f = sample["fields"]
    arr = (f["ux"] ** 2 + f["uy"] ** 2).sqrt() if name == "speed" else f[name]
    arr = arr.numpy().astype(np.float64)
    return np.where(f["mask"].numpy(), np.nan, arr)


def _render_one(sample: dict, out_dir: Path, cfg) -> Path:
    """渲染单样本：len(fields) 个横排面板 + 参数标题，存 sample_%08d.png。"""
    p = sample["params"]
    n = len(cfg.vis.fields)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), cfg.vis.fields):
        im = ax.imshow(_panel_array(sample, name), origin="lower", cmap=cfg.vis.cmap)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"#{sample['index']}  Re={p['reynolds_lattice']:.0f}  Ma={p['mach_lattice']:.3f}"
                 f"  AoA={p['aoa_deg']:.1f}°  NACA m={p['naca_m']:.3f} p={p['naca_p']:.2f}"
                 f" t={p['naca_t']:.3f}  steps={p['steps']}")
    path = out_dir / f"sample_{sample['index']:08d}.png"
    fig.savefig(path, dpi=cfg.vis.dpi)
    plt.close(fig)
    return path


def render_samples(cfg, indices: list | None = None) -> list:
    """渲染稳态样本为 PNG。

    参数:
        cfg: 配置对象，读取 cfg.storage.path 与 cfg.vis.*
        indices: 指定样本编号列表（可含断点空洞中的任意编号）；
                 None 时渲染前 vis.num_samples 个（0=全部）
    返回:
        渲染生成的 PNG 路径列表（按样本索引升序）；不存在的编号跳过并告警
    """
    ds = FlowFieldDataset(cfg.storage.path)
    check_render_inputs(ds, cfg.vis)
    out_dir = Path(cfg.vis.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if indices is None:
        n = len(ds) if cfg.vis.num_samples == 0 else min(cfg.vis.num_samples, len(ds))
        samples = [ds[i] for i in range(n)]
    else:
        samples = []
        for i in sorted(set(indices)):
            s = ds.get_by_index(i)
            if s is None:
                print(f"[vis] 警告：样本 {i} 不存在（可能未收敛被丢弃），跳过")
                continue
            samples.append(s)
    paths = [_render_one(s, out_dir, cfg) for s in samples]
    print(f"[vis] 已渲染 {len(paths)} 个样本 → {out_dir}")
    return paths
