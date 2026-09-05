"""带二维旋转位置编码的 12 层静态注意力残差 Transformer 与三级 FP32 解码器

模块: model/flow_transformer/flow_transformer.py
依赖: torch, model.flow_transformer.checks
读取配置: grid.nx, grid.ny, model.*
对外接口:
    - RMSNorm: 最后一维均方根归一化
    - RMSNorm2d: 图像通道维均方根归一化
    - FlowResidualTransformer: 输入面元基线场并预测 ux/uy/p 归一化残差
说明: 嵌入、二维 RoPE 旋转与完整解码器固定 FP32；Transformer 主干服从外层 autocast。
      静态 AttnRes 保存各子层原始增量，绝不执行普通残差加回。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.flow_transformer.checks import check_model_inputs

__all__ = ["RMSNorm", "RMSNorm2d", "FlowResidualTransformer"]


class RMSNorm(nn.Module):
    """在最后一维执行 RMSNorm，归一化统计始终用 FP32 累积。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


class RMSNorm2d(nn.Module):
    """在二维特征图的通道维执行 RMSNorm，不混合空间统计。"""

    def __init__(self, channels: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + self.eps)
        return (normalized * self.weight.float().view(1, -1, 1, 1)).to(dtype)


class _RoPE2D(nn.Module):
    def __init__(self, height: int, width: int, head_dim: int, base: float):
        super().__init__()
        axis_dim = head_dim // 2
        frequencies = torch.exp(torch.arange(0, axis_dim, 2, dtype=torch.float32)
                                * (-math.log(base) / axis_dim))
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        for axis, positions in (("y", y), ("x", x)):
            angles = positions.reshape(-1, 1) * frequencies
            self.register_buffer(f"cos_{axis}", angles.cos()[None, None], persistent=False)
            self.register_buffer(f"sin_{axis}", angles.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_axis(values: torch.Tensor, cosine: torch.Tensor,
                     sine: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        real, imaginary = values.float().reshape(*values.shape[:-1], -1, 2).unbind(dim=-1)
        rotated = torch.stack((real * cosine - imaginary * sine,
                               imaginary * cosine + real * sine), dim=-1)
        return rotated.flatten(-2).to(dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        y, x = values.chunk(2, dim=-1)
        return torch.cat((self._rotate_axis(y, self.cos_y, self.sin_y),
                          self._rotate_axis(x, self.cos_x, self.sin_x)), dim=-1)


class _SpatialAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: _RoPE2D) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class _SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.expand = nn.Linear(dim, hidden)
        self.project = nn.Linear(hidden // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.expand(x).chunk(2, dim=-1)
        return self.project(value * F.silu(gate))


class _TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.attn_norm = RMSNorm(m.dim, m.norm_eps)
        self.attn = _SpatialAttention(m.dim, m.heads, m.dropout)
        self.ffn_norm = RMSNorm(m.dim, m.norm_eps)
        self.ffn = _SwiGLUFeedForward(m.dim, m.ffn_hidden)


def _icnr_(weight: torch.Tensor, scale: int = 2) -> None:
    """让 PixelShuffle 的各子像素相位共享初始卷积核，抑制棋盘格。"""
    out_channels, in_channels, kh, kw = weight.shape
    base = torch.empty(out_channels // (scale * scale), in_channels, kh, kw,
                       device=weight.device, dtype=weight.dtype)
    nn.init.kaiming_normal_(base)
    with torch.no_grad():
        weight.copy_(base.repeat_interleave(scale * scale, dim=0))


class _RefineBlock(nn.Module):
    def __init__(self, channels: int, eps: float):
        super().__init__()
        self.norm = RMSNorm2d(channels, eps)
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.project = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.norm(x))
        value, gate = self.expand(y).chunk(2, dim=1)
        return x + self.project(value * F.silu(gate))


class _PixelShuffleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float):
        super().__init__()
        self.shuffle_conv = nn.Conv2d(in_channels, 4 * out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = _RefineBlock(out_channels, eps)
        _icnr_(self.shuffle_conv.weight)
        nn.init.zeros_(self.shuffle_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = F.pixel_shuffle(self.shuffle_conv(x), 2)
        shortcut = self.skip(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))
        return self.refine((learned + shortcut) * (2.0 ** -0.5))


class FlowResidualTransformer(nn.Module):
    """由 SDF、面元初值和工况条件预测稳态黏性修正。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m, g = cfg.model, cfg.grid
        token_h, token_w = g.ny // m.patch_size, g.nx // m.patch_size
        self.patch_embed = nn.Conv2d(m.input_channels, m.dim, m.patch_size, stride=m.patch_size)
        self.condition_embed = nn.Linear(3, m.dim)
        self.rope = _RoPE2D(token_h, token_w, m.dim // m.heads, m.rope_base)
        self.blocks = nn.ModuleList([_TransformerBlock(cfg) for _ in range(m.depth)])
        routes = 2 * m.depth + 1
        self.routing_logits = nn.Parameter(torch.zeros(routes * (routes + 1) // 2))
        self.final_norm = RMSNorm(m.dim, m.norm_eps)
        self.decoder_stem = nn.Conv2d(m.dim, m.dim, 3, padding=1)
        channels = [m.dim, *m.decoder_channels]
        self.decoder = nn.ModuleList([
            _PixelShuffleStage(channels[i], channels[i + 1], m.norm_eps) for i in range(3)
        ])
        self.output_norm = RMSNorm2d(channels[-1], m.norm_eps)
        self.output = nn.Conv2d(channels[-1], m.output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _mix(self, values: list[torch.Tensor], row: int) -> torch.Tensor:
        stacked = torch.stack(values, dim=0)
        start = row * (row + 1) // 2
        weights = self.routing_logits[start:start + len(values)].float().softmax(dim=0)
        return torch.einsum("n,nbtd->btd", weights.to(stacked.dtype), stacked)

    def _embedding(self, fields: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        device_type = fields.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            embedded = self.patch_embed(fields.float()).flatten(2).transpose(1, 2)
            return embedded + self.condition_embed(conditions.float()).unsqueeze(1)

    def _decoder(self, tokens: torch.Tensor) -> torch.Tensor:
        device_type = tokens.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            batch = tokens.shape[0]
            h = self.cfg.grid.ny // self.cfg.model.patch_size
            w = self.cfg.grid.nx // self.cfg.model.patch_size
            x = self.final_norm(tokens.float()).transpose(1, 2).reshape(batch, self.cfg.model.dim, h, w)
            x = self.decoder_stem(x)
            for stage in self.decoder:
                x = stage(x)
            return self.output(self.output_norm(x))

    def forward(self, fields: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """预测 `(B,3,H,W)` 的通道尺度化残差；返回值始终为 FP32。"""
        check_model_inputs(fields, conditions, self.cfg)
        tokens = self._embedding(fields, conditions)
        if fields.is_cuda and torch.is_autocast_enabled():
            tokens = tokens.to(torch.get_autocast_gpu_dtype())
        values = [tokens]
        row = 0
        for block in self.blocks:
            attn_input = self._mix(values, row)
            values.append(block.attn(block.attn_norm(attn_input), self.rope))
            row += 1
            ffn_input = self._mix(values, row)
            values.append(block.ffn(block.ffn_norm(ffn_input)))
            row += 1
        return self._decoder(self._mix(values, row))

    @torch.no_grad()
    def routing_weights(self) -> torch.Tensor:
        """返回零填充的因果深度路由权重，每行有效部分之和为 1。"""
        routes = 2 * self.cfg.model.depth + 1
        result = self.routing_logits.new_zeros(routes, routes)
        for row in range(routes):
            start = row * (row + 1) // 2
            result[row, :row + 1] = self.routing_logits[start:start + row + 1].softmax(dim=0)
        return result
