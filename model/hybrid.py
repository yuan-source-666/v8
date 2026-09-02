"""混合骨干层：Mamba-2 与 Self-Attention 按 2:1 交替，pre-norm，含 SwiGLU 前馈。

对齐架构文档 §1.2 / §1.3：
  · RMSNorm —— 现代小模型事实标准、省算力、稳定（pre-norm 残差结构）
  · SwiGLU —— 门控前馈，同等参数量表达更强
  · 布局   —— 如 "MMA" 表示每 3 层一组 [Mamba, Mamba, Attention]
              （6 层 → M M A M M A，即 4 Mamba + 2 Attention，Mamba 占比 2/3）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RMSNorm", "SwiGLUFFN", "HybridBlock", "HybridBackbone", "build_layout"]


class RMSNorm(nn.Module):
    """RMS 归一化（无均值平移，仅缩放），pre-norm 布局使用。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU 门控前馈：down( SiLU(gate(x)) * up(x) )。"""

    def __init__(self, d_model, ffn_mult=4, bias=False):
        super().__init__()
        inter = int(ffn_mult * d_model)
        self.gate = nn.Linear(d_model, inter, bias=bias)
        self.up = nn.Linear(d_model, inter, bias=bias)
        self.down = nn.Linear(inter, d_model, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def build_layout(n_layer, layout_str):
    """把布局字符串循环展开为每层类型列表。

    >>> build_layout(6, "MMA")
    ['m', 'm', 'a', 'm', 'm', 'a']
    """
    types = []
    n = len(layout_str)
    for i in range(n_layer):
        types.append(layout_str[i % n].lower())
    return types


class HybridBlock(nn.Module):
    """单个混合块：pre-norm 残差 →（Mamba-2 或 Attention 子层）→ pre-norm 残差 → SwiGLU。"""

    def __init__(self, d_model, sublayer, ffn):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.sublayer = sublayer            # Mamba2 或 Attention
        self.norm2 = RMSNorm(d_model)
        self.ffn = ffn

    def forward(self, x):
        x = x + self.sublayer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridBackbone(nn.Module):
    """由多个 HybridBlock 组成的混合骨干（按 layout 交替 Mamba-2 / Attention）。"""

    def __init__(self, d_model, n_layer, layout_str, ffn_mult,
                 mamba_cfg, attn_cfg):
        super().__init__()
        types = build_layout(n_layer, layout_str)
        blocks = []
        for idx, t in enumerate(types):
            if t == "m":
                from .mamba2 import Mamba2
                sub = Mamba2(d_model, **mamba_cfg)
            elif t == "a":
                from .attention import Attention
                sub = Attention(d_model, **attn_cfg)
            else:
                raise ValueError(f"layout 中未知类型: {t!r}")
            ffn = SwiGLUFFN(d_model, ffn_mult=ffn_mult)
            blocks.append(HybridBlock(d_model, sub, ffn))
        self.blocks = nn.ModuleList(blocks)
        self.types = types

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    from .mamba2 import Mamba2
    from .attention import Attention
    m = HybridBackbone(
        d_model=64, n_layer=6, layout_str="MMA", ffn_mult=4,
        mamba_cfg=dict(expand=2, d_state=16, d_conv=4, headdim=32, dt=True),
        attn_cfg=dict(n_head=4, n_kv_head=1, head_dim=16, max_seq_len=64),
    )
    x = torch.randn(2, 32, 64)
    y = m(x)
    y.mean().backward()
    print("HybridBackbone OK:", tuple(y.shape), "layout=", "".join(m.types))
