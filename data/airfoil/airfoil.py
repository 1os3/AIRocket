"""参数化翼型生成与 GPU 栅格化：NACA 四位数族 → 反弹边界掩码与有符号距离场

模块: data/airfoil/airfoil.py
依赖: torch, data.airfoil.checks.airfoil_checks
读取配置: grid.nx, grid.ny, grid.chord, grid.x_le, grid.y_center, airfoil.n_points
对外接口:
    - naca4_polygon(m, p, t, n_points, device) -> (K, 2) 张量
    - build_airfoil_polygon(cfg, m, p, t, aoa_deg, device) -> (K, 2) 张量
    - build_airfoil_geometry(cfg, m, p, t, aoa_deg, device) -> (mask, sdf)
    - build_airfoil_mask(cfg, m, p, t, aoa_deg, device) -> (ny, nx) bool 张量
说明: 栅格化用 torch 向量化的射线法（even-odd 规则），全程在目标设备上计算，
      不向 CPU 回传；sdf 由点到边界折线的最近距离给出（固体内取负），
      是 Bouzidi 插值反弹估计壁面分数 q 的依据。
"""

import math

import torch

from data.airfoil.checks.airfoil_checks import check_naca_params

__all__ = ["naca4_polygon", "build_airfoil_polygon", "build_airfoil_geometry", "build_airfoil_mask"]


def naca4_polygon(m: float, p: float, t: float, n_points: int, device) -> torch.Tensor:
    """生成 NACA 四位数翼型闭合多边形（弦长归一，前缘在原点）。

    参数:
        m: 最大弯度（弦长分数）；p: 弯度位置；t: 最大厚度；n_points: 半面取样数
    返回:
        (2*n_points, 2) 张量：上表面前缘→后缘，再下表面后缘→前缘
    """
    check_naca_params(m, p, t)
    beta = torch.linspace(0.0, math.pi, n_points, device=device)
    x = 0.5 * (1.0 - torch.cos(beta))  # 余弦加密：前缘/后缘点更密
    yt = 5.0 * t * (0.2969 * torch.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)  # −0.1036 使后缘闭合
    front = x < p  # 弯度线分段：m==0 时公式退化为 0，无需分支
    yc = torch.where(front, m / p ** 2 * (2.0 * p * x - x ** 2),
                     m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2))
    dyc = torch.where(front, 2.0 * m / p ** 2 * (p - x), 2.0 * m / (1.0 - p) ** 2 * (p - x))
    theta = torch.atan(dyc)
    upper = torch.stack([x - yt * torch.sin(theta), yc + yt * torch.cos(theta)], dim=1)
    lower = torch.stack([x + yt * torch.sin(theta), yc - yt * torch.cos(theta)], dim=1)
    return torch.cat([upper, lower.flip(0)[1:]], dim=0)  # 去掉重复的前缘点


def _rotate(points: torch.Tensor, aoa_deg: float) -> torch.Tensor:
    """绕 (0.25, 0)（1/4 弦点，近似气动中心）按攻角逆时针旋转。"""
    rad = math.radians(aoa_deg)
    c, s = math.cos(rad), math.sin(rad)
    center = torch.tensor([0.25, 0.0], device=points.device)
    rot = torch.tensor([[c, -s], [s, c]], device=points.device)
    return (points - center) @ rot.T + center


def build_airfoil_polygon(cfg, m: float, p: float, t: float,
                          aoa_deg: float, device) -> torch.Tensor:
    """生成旋转后的弦长归一翼型多边形，供栅格化和面元初始化共用。"""
    return _rotate(naca4_polygon(m, p, t, cfg.airfoil.n_points, device), aoa_deg)


def _points_in_polygon(xs: torch.Tensor, ys: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """向量化射线法：返回各点是否在多边形内（even-odd 规则）。

    参数: xs/ys 为一维展平坐标，poly 为 (K,2) 闭合前顶点序列
    """
    x1, y1 = poly[:, 0], poly[:, 1]
    x2 = torch.roll(x1, -1)
    y2 = torch.roll(y1, -1)  # 末点与首点闭合
    crosses = ((y1 > ys[:, None]) != (y2 > ys[:, None])) & (
        xs[:, None] < (x2 - x1) * (ys[:, None] - y1) / (y2 - y1) + x1
    )
    return crosses.sum(dim=1) % 2 == 1


def _signed_distance(xs: torch.Tensor, ys: torch.Tensor, poly: torch.Tensor,
                     inside: torch.Tensor) -> torch.Tensor:
    """点到边界折线的最近距离，多边形内取负（供 Bouzidi 估计壁面分数 q）。

    参数: xs/ys 一维展平坐标；inside 为同形状 bool（True=在多边形内）
    返回: 与 xs 同形状的有符号距离（弦长单位）
    """
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = torch.roll(x1, -1), torch.roll(y1, -1)
    dx, dy = x2 - x1, y2 - y1
    seg_len2 = (dx * dx + dy * dy).clamp_min(1e-12)
    chunk = 1 << 14  # 距离场是浮点中间量，块比布尔射线法再小一档
    dist2 = torch.full_like(xs, float("inf"))
    for i in range(0, xs.numel(), chunk):
        # 投影参数截断到 [0,1]，最近点钳在线段上
        u = (((xs[i:i + chunk, None] - x1) * dx + (ys[i:i + chunk, None] - y1) * dy)
             / seg_len2).clamp(0.0, 1.0)
        d2 = (xs[i:i + chunk, None] - (x1 + u * dx)) ** 2 + (ys[i:i + chunk, None] - (y1 + u * dy)) ** 2
        dist2[i:i + chunk] = d2.min(dim=1).values
    sdf = dist2.sqrt()
    sdf[inside] = -sdf[inside]
    return sdf


def build_airfoil_geometry(cfg, m: float, p: float, t: float, aoa_deg: float, device) -> tuple:
    """把采样翼型映射到 LBM 网格，生成固体掩码与有符号距离场。

    返回:
        (mask, sdf)：mask 为 (ny, nx) bool（True=固体）；
        sdf 为 (ny, nx) 浮点，流体内为正、固体内为负的边界距离（格子单位），
        供 Bouzidi 插值反弹估计壁面分数 q
    """
    g = cfg.grid
    poly = build_airfoil_polygon(cfg, m, p, t, aoa_deg, device)
    # 网格物理坐标（弦长归一）：翼型前缘置于 (x_le, y_center)，y 轴向上为正
    xs = (torch.arange(g.nx, device=device, dtype=poly.dtype) + 0.5 - g.x_le) / g.chord
    ys = (torch.arange(g.ny, device=device, dtype=poly.dtype) + 0.5 - g.y_center) / g.chord
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    flat_x, flat_y = xx.reshape(-1), yy.reshape(-1)
    chunk = 1 << 16  # 分块限制 (N×K) 中间布尔张量的显存占用
    inside = torch.cat([_points_in_polygon(flat_x[i:i + chunk], flat_y[i:i + chunk], poly)
                        for i in range(0, flat_x.numel(), chunk)])
    sdf = _signed_distance(flat_x, flat_y, poly, inside) * float(g.chord)  # 换成格子单位
    return inside.reshape(g.ny, g.nx), sdf.reshape(g.ny, g.nx)


def build_airfoil_mask(cfg, m: float, p: float, t: float, aoa_deg: float, device) -> torch.Tensor:
    """参数化翼型 → LBM 固体节点掩码（build_airfoil_geometry 的掩码部分）。

    返回:
        (ny, nx) bool 张量，True 为翼型固体节点
    """
    return build_airfoil_geometry(cfg, m, p, t, aoa_deg, device)[0]
