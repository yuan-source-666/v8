"""Mamba-2 SSD 层 —— 纯 PyTorch 实现（简化版）。

参考：state-spaces/mamba（https://github.com/state-spaces/mamba）
语义要点（对齐 Mamba-2 / Nemotron Nano 2 的替换层设计）：
  · 输入门（input gate z）+ 输出门（output gate，经 SiLU(z) 调制）
  · 因果卷积（causal depthwise conv1d：左侧补零保证因果，不使用未来信息）
  · 选择性状态空间：B、C 由当前输入 x 经 x_proj 投影产生（selective），
    A 为每 head 可学习的收缩系数（标量对角阵，位于 (0,1)）
  · 状态空间双对偶（SSD quadratic form）：把对角线性 RNN 改写为显式求和
        y_t = Σ_{s<=t} α^{t-s} (C_t · B_s) x_s
    用 einsum + 因果衰减掩码一次性向量化（等价于逐时间步的线性 scan，
    但在 CPU 上得益于批量矩阵运算，吞吐远高于 while 循环）。

为什么用显式 SSD 公式：Mamba-2 论文的核心正是把注意力的二次形式与
线性 RNN 的状态递归统一（State Space Duality），本实现取其矩阵化分支。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Mamba2", "ssd_scan"]


def ssd_scan(x, B, C, alpha):
    """显式 SSD（状态空间双对偶 / quadratic form）扫描。

    Args:
        x:     (B, L, H, D)  主分支（head 化后的输入）
        B:     (B, L, H, S)  选择性输入矩阵
        C:     (B, L, H, S)  选择性输出矩阵
        alpha: (H,)          每 head 收缩系数 α ∈ (0,1)
    Returns:
        y: (B, L, H, D)      y_t = Σ_{s<=t} α^{t-s} (C_t·B_s) x_s
    """
    Bs, L = x.shape[0], x.shape[1]
    # 相似度矩阵 S[t,s] = C_t · B_s  (B, H, L, L)
    S = torch.einsum("b t h s, b m h s -> b h t m", C, B)
    # 因果衰减矩阵 Decay[t,s] = α^{t-s} (t>=s)，用 log 空间稳定计算
    idx = torch.arange(L, device=x.device)
    rel = idx[None, :] - idx[:, None]                      # (L, L)
    causal = (rel >= 0)
    log_alpha = torch.log(alpha.clamp(min=1e-12))          # (H,)
    decay_val = torch.exp(log_alpha[:, None, None] * rel.clamp(min=0))  # (H, L, L)
    decay = torch.where(causal[None], decay_val, torch.zeros_like(decay_val))
    A = S * decay                                          # (B, H, L, L)
    del S, decay_val
    y = torch.einsum("b h t m, b m h d -> b t h d", A, x)  # (B, L, H, D)
    return y


class Mamba2(nn.Module):
    """Mamba-2 SSD 层（简化版，可作 Transformer 自注意力的替换层）。

    配置项见 config/mini.yaml 的 `mamba:` 段：
      expand   d_inner = expand * d_model
      d_state  状态维度 S
      d_conv   因果卷积核大小
      headdim  内部分头维度 → H = d_inner // headdim
      dt       是否启用可学习时间步缩放（并入 decay 指数）
    """

    def __init__(self, d_model, expand=2, d_state=16, d_conv=4,
                 headdim=64, dt=True, bias=False, eps=1e-5):
        super().__init__()
        assert d_model > 0
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, "d_inner 必须能被 headdim 整除"
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt = dt

        # 输入投影 → (z | x)，z 为输出门，x 为主分支
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        # 选择性投影 → (B | C)，每 head d_state 维
        self.x_proj = nn.Linear(self.d_inner, self.nheads * d_state * 2, bias=bias)

        # 收缩系数 A：每 head 标量对角，alpha = exp(-exp(A_log)) ∈ (0,1) 恒正
        self.A_log = nn.Parameter(torch.randn(self.nheads) * 0.5 - 1.0)
        # 可学习时间步缩放（并入 decay 指数；`dt=True` 时才有）
        if dt:
            self.dt_log = nn.Parameter(torch.randn(self.nheads) * 0.1 - 2.5)
        else:
            self.register_buffer("dt_log", None)

        # 因果卷积（depthwise，通道独立卷积）
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, padding=0,
                                groups=self.d_inner, bias=bias)
        self.act = nn.SiLU()
        from .hybrid import RMSNorm  # 延迟导入，避免顶层循环依赖
        self.norm = RMSNorm(self.d_inner, eps=eps)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _alpha(self):
        """每 head 收缩系数 α ∈ (0,1)。"""
        logit = self.A_log
        if self.dt_log is not None:
            logit = logit + self.dt_log          # dt 并入 decay 速率
        return torch.exp(-torch.exp(logit))

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        zx = self.in_proj(x)                       # (B, L, 2*d_inner)
        z, xr = torch.chunk(zx, 2, dim=-1)         # 各 (B, L, d_inner)

        # 因果卷积（主分支），然后 SiLU
        xr = xr.transpose(1, 2)                    # (B, d_inner, L)
        xr = F.pad(xr, (self.d_conv - 1, 0))
        xr = self.conv1d(xr)                       # (B, d_inner, L)
        xr = self.act(xr.transpose(1, 2))          # (B, L, d_inner)

        # head 化
        xr = xr.view(B, L, self.nheads, self.headdim)   # (B, L, H, D)

        # 选择性 B、C
        bc = self.x_proj(xr.reshape(B, L, self.d_inner))   # (B, L, H*2S)
        bc = bc.view(B, L, self.nheads, 2 * self.d_state)
        Bm, Cm = torch.split(bc, self.d_state, dim=-1)

        # SSD 扫描（矩阵化二次形式）
        y = ssd_scan(xr, Bm, Cm, self._alpha())    # (B, L, H, D)
        y = y.reshape(B, L, self.d_inner)

        # 输出门：RMSNorm 后乘 SiLU(z) 调制
        y = self.norm(y) * self.act(z)
        return self.out_proj(y)                    # (B, L, d_model)


if __name__ == "__main__":
    # 单元自检：前向 + 反向一步
    torch.manual_seed(0)
    m = Mamba2(d_model=64, expand=2, d_state=16, d_conv=4, headdim=32)
    x = torch.randn(2, 32, 64)
    y = m(x)
    y.mean().backward()
    print("Mamba2 OK:", tuple(y.shape), "params(M)=", sum(p.numel() for p in m.parameters()) / 1e6)
