"""D2Q9 多松弛时间（MRT）格子玻尔兹曼批量求解器，全流程在 GPU 张量上完成

模块: data/lbm_solver/lbm_solver.py
依赖: torch, data.lbm_solver.checks.lbm_solver_checks
读取配置: grid.nx, grid.ny, solver.boundary, solver.max_steps, solver.check_interval,
          solver.conv_tol, solver.float64, solver.mrt_s
对外接口:
    - LBMSolver: 批量求解器；run_batch(masks, u_lb, tau, sdfs=None)
      -> dict(rho, ux, uy, p, mask, steps, converged)
说明:
    - 分布函数 f 形状 (B, 9, H, W)：dim0 批样本、dim1 离散速度方向、dim2=y、dim3=x。
    - 上下边界由 torch.roll 天然周期化；左边界 Zou-He 速度入口、右边界零梯度出口。
    - 固壁两种格式（solver.boundary）：bounce_back 全反弹；bouzidi 插值反弹
      （Bouzidi-Firdaouss-Lallemand 2001，按 SDF 估计的壁面分数 q 分 q<1/2 与
      q≥1/2 两式二次插值，链路掩码与 q 每批预计算一次，步内只剩逐方向 roll）。
    - MRT 碰撞折叠为批矩阵 A = M⁻¹·S·M（每样本 tau 不同 → A 形状 (B,9,9)）。
    - 批内各样本独立判定收敛/发散；已完成样本冻结（不再更新），整批一次迭代循环。
    - 每步迭代无 CPU-GPU 往返；仅收敛判定周期性地取标量，最终结果一次性回传 CPU。
"""

import torch

from data.lbm_solver.checks.lbm_solver_checks import check_run_inputs

# D2Q9 离散速度（数学常量，非配置）：0 静止，1-4 主轴，5-8 对角
_EX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
_EY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
_W = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
_OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # 各方向的反弹对偶方向
_CS2 = 1.0 / 3.0  # 格子声速平方

# Lallemand-Luo (2000) D2Q9 矩变换矩阵：行依次为 ρ,e,ε,jx,qx,jy,qy,pxx,pxy
_M = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1],
]

__all__ = ["LBMSolver", "CS2"]

CS2 = _CS2


class LBMSolver:
    """批量稳态求解器：构造一次，按批调用 run_batch()。"""

    def __init__(self, cfg, device: torch.device):
        self._cfg = cfg
        self._device = device
        dtype = torch.float64 if cfg.solver.float64 else torch.float32
        self._dtype = dtype
        ny, nx = cfg.grid.ny, cfg.grid.nx
        self._ex = torch.tensor(_EX, dtype=dtype, device=device).view(1, 9, 1, 1)
        self._ey = torch.tensor(_EY, dtype=dtype, device=device).view(1, 9, 1, 1)
        self._w = torch.tensor(_W, dtype=dtype, device=device).view(1, 9, 1, 1)
        m = torch.tensor(_M, dtype=dtype, device=device)
        self._m_inv = torch.linalg.inv(m)
        self._m = m
        self._solid = torch.zeros(1, 1, ny, nx, dtype=torch.bool, device=device)  # run_batch 时绑定

    def _equilibrium(self, rho, ux, uy):
        """平衡分布：rho (B,1,H,W)，ux/uy (B,H,W) → feq (B,9,H,W)。"""
        eu = self._ex * ux.unsqueeze(1) + self._ey * uy.unsqueeze(1)
        u2 = (ux * ux + uy * uy).unsqueeze(1)
        return self._w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)

    def _collide(self, f, feq, a):
        """MRT 碰撞：f* = f − A·(f − feq)，A (B,9,9) 由 run_batch() 一次性构建。"""
        df = (f - feq).reshape(f.shape[0], 9, -1)
        return f - (a @ df).reshape_as(f)

    def _stream(self, f):
        """迁移：按离散速度方向整体平移；roll 的环绕即上下周期边界。"""
        return torch.stack([torch.roll(f[:, i], shifts=(int(_EY[i]), int(_EX[i])), dims=(1, 2))
                            for i in range(9)], dim=1)

    def _apply_bc(self, f, u_lb):
        """边界处理：固体反弹 + 左 Zou-He 速度入口 + 右零梯度出口。u_lb 为 (B,)。"""
        f = torch.where(self._solid, f[:, _OPP], f)  # 固壁简单反弹（半反弹精度的全反弹形式）
        # Zou-He 入口（x=0 列）：由已知族群反解密度并补齐东向族群 f1,f5,f8
        u = u_lb.view(-1, 1)
        rho_in = (f[:, 0, :, 0] + f[:, 2, :, 0] + f[:, 4, :, 0]
                  + 2.0 * (f[:, 3, :, 0] + f[:, 6, :, 0] + f[:, 7, :, 0])) / (1.0 - u)
        f[:, 1, :, 0] = f[:, 3, :, 0] + (2.0 / 3.0) * rho_in * u
        f[:, 5, :, 0] = f[:, 7, :, 0] + 0.5 * (f[:, 4, :, 0] - f[:, 2, :, 0]) + rho_in * u / 6.0
        f[:, 8, :, 0] = f[:, 6, :, 0] + 0.5 * (f[:, 2, :, 0] - f[:, 4, :, 0]) + rho_in * u / 6.0
        f[:, :, :, -1] = f[:, :, :, -2]  # 出口零梯度
        return f

    def _macro(self, f):
        """宏观量：rho (B,1,H,W)，ux/uy (B,H,W)；固体内速度强制为零。"""
        rho = f.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ux = (f * self._ex).sum(dim=1) / rho.squeeze(1)
        uy = (f * self._ey).sum(dim=1) / rho.squeeze(1)
        solid = self._solid.squeeze(1)
        return rho, ux.masked_fill(solid, 0.0), uy.masked_fill(solid, 0.0)

    def _setup_bouzidi(self, masks: torch.Tensor, sdfs: torch.Tensor) -> list:
        """预计算每条固-流链路的边界掩码与壁面分数 q（线性近似 q=df/(df−ds)）。"""
        solid = masks.to(self._device)
        sdf = sdfs.to(device=self._device, dtype=self._dtype)
        links = []
        for i in range(1, 9):  # 0 为静止方向，无链路
            ex, ey = int(_EX[i]), int(_EY[i])
            ns = torch.roll(solid, shifts=(-ey, -ex), dims=(1, 2))  # x+ci 处是否固体
            bmask = ~solid & ns                                     # 流→固链路的流体端
            ss = torch.roll(sdf, shifts=(-ey, -ex), dims=(1, 2))    # 固体端距离（负）
            q = (sdf / (sdf - ss)).clamp(1e-3, 1.0 - 1e-3)
            links.append((i, _OPP[i], (ex, ey), bmask, q))
        return links

    def _apply_bouzidi(self, f_new, f_star, links, fluid):
        """Bouzidi 插值反弹：改写边界流体节点来自固方向的族群 f[ī]。

        q<1/2 用上游两流体节点二次插值；q≥1/2 的 f_ī 记忆项取碰撞后族群
        （原式的递归存储值会与 MRT 形成正反馈、数百步内发散，实测取 f* 变体
        稳定且保持二阶精度）；上游节点不足（薄翼尖/前缘尖点）时回退简单反弹。
        """
        for i, ib, (ex, ey), bmask, q in links:
            fi_x = f_star[:, i]                                     # f_i*(xf)
            fi_xm = torch.roll(f_star[:, i], shifts=(ey, ex), dims=(1, 2))   # f_i*(xf−ci)
            fi_xmm = torch.roll(f_star[:, i], shifts=(2 * ey, 2 * ex), dims=(1, 2))
            fib_m = torch.roll(f_star[:, ib], shifts=(ey, ex), dims=(1, 2))  # f_ī*(xf−ci)
            ok1 = torch.roll(fluid, shifts=(ey, ex), dims=(1, 2))   # xf−ci 是否流体
            ok2 = ok1 & torch.roll(fluid, shifts=(2 * ey, 2 * ex), dims=(1, 2))
            small = q < 0.5
            val_small = q * (1.0 + 2.0 * q) * fi_x + (1.0 - 4.0 * q * q) * fi_xm \
                - q * (1.0 - 2.0 * q) * fi_xmm
            val_big = fi_x / (q * (2.0 * q + 1.0)) + ((2.0 * q - 1.0) / q) * f_star[:, ib] \
                + ((1.0 - 2.0 * q) / (1.0 + 2.0 * q)) * fib_m
            val_small = torch.where(ok2, val_small, fi_x)
            val_big = torch.where(ok1, val_big, fi_x)
            f_new[:, ib] = torch.where(bmask, torch.where(small, val_small, val_big), f_new[:, ib])
        return f_new

    def run_batch(self, masks: torch.Tensor, u_lb: torch.Tensor, tau: torch.Tensor,
                  sdfs: torch.Tensor | None = None) -> dict:
        """并行求解一批工况至各自稳态。

        参数:
            masks: (B, H, W) bool 张量，True 为翼型固体节点（反弹边界）
            u_lb: (B,) 格子单位来流速度（由 Ma 换算并经截断）
            tau: (B,) 松弛时间，nu = (tau − 0.5)·cs²
            sdfs: (B, H, W) 有符号距离场（流体内正、固体内负）；boundary=bouzidi 时必传
        返回:
            dict(rho, ux, uy, p, mask, steps, converged)：场量为 (B,H,W) CPU 张量，
            steps/converged 为长度 B 的列表；converged=False 含发散/超限两种失败
        """
        check_run_inputs(masks, u_lb, tau, self._cfg.grid, sdfs, self._cfg.solver.boundary)
        b = masks.shape[0]
        links = self._setup_bouzidi(masks, sdfs) if self._cfg.solver.boundary == "bouzidi" else None
        self._solid = masks.to(self._device).unsqueeze(1)
        one = torch.ones(b, 1, *masks.shape[1:], dtype=self._dtype, device=self._device)
        zero = torch.zeros(b, *masks.shape[1:], dtype=self._dtype, device=self._device)
        f = self._equilibrium(one, u_lb.view(b, 1, 1).expand_as(zero).contiguous(), zero)
        # 碰撞矩阵批内固定，只构建一次（tau 逐样本不同 → (B,9,9)）
        s = torch.cat([torch.tensor(self._cfg.solver.mrt_s, dtype=self._dtype,
                                    device=self._device).expand(b, 7),
                       (1.0 / tau).unsqueeze(1).expand(b, 2)], dim=1)
        a = self._m_inv.unsqueeze(0) @ torch.diag_embed(s) @ self._m.unsqueeze(0)
        done = torch.zeros(b, dtype=torch.bool, device=self._device)      # 收敛或发散
        failed = torch.zeros(b, dtype=torch.bool, device=self._device)    # 仅发散
        steps = torch.zeros(b, dtype=torch.long, device=self._device)
        prev = None
        for step in range(1, self._cfg.solver.max_steps + 1):
            rho, ux, uy = self._macro(f)
            f_col = self._collide(f, self._equilibrium(rho, ux, uy), a)
            f_new = self._apply_bc(self._stream(f_col), u_lb)
            if links is not None:
                f_new = self._apply_bouzidi(f_new, f_col, links, ~self._solid.squeeze(1))
            f = torch.where(done.view(b, 1, 1, 1), f, f_new)  # 已完成样本冻结
            if step % self._cfg.solver.check_interval == 0:
                rho, ux, uy = self._macro(f)
                nan = torch.isnan(rho).any(dim=(1, 2, 3)) & ~done  # (B,)
                steps = torch.where(nan & (steps == 0), torch.full_like(steps, step), steps)
                failed |= nan
                done |= nan
                if prev is not None:
                    # 稳态判据用全场 L2 相对变化：max 范数会被壁面单点的 f32 级
                    # 残余振荡卡在 ~1e-5（Bouzidi 尤为明显），L2 对单点噪声鲁棒
                    num = ((ux - prev[0]) ** 2 + (uy - prev[1]) ** 2).sum(dim=(1, 2)).sqrt()
                    den = (ux ** 2 + uy ** 2).sum(dim=(1, 2)).sqrt()
                    hit = (num / den.clamp_min(1e-12) < self._cfg.solver.conv_tol) & ~done
                    steps = torch.where(hit, torch.full_like(steps, step), steps)
                    done |= hit
                prev = (ux.clone(), uy.clone())
                if bool(done.all()):  # 整批结束，提前退出
                    break
        steps = torch.where(steps == 0, torch.full_like(steps, step), steps)
        converged = done & ~failed
        # 仅在批结束时一次性回传 CPU；压力取 p = cs²·(ρ − 1)（去静压）
        rho2d = rho.squeeze(1)
        fields = {k: v.detach().to("cpu", torch.float32)
                  for k, v in {"rho": rho2d, "ux": ux, "uy": uy,
                               "p": (rho2d - 1.0) * _CS2}.items()}
        return {**fields, "mask": self._solid.squeeze(1).cpu(),
                "steps": steps.tolist(), "converged": converged.tolist()}
