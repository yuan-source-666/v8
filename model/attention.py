"""GQA（分组查询注意力）+ RoPE（旋转位置编码）自注意力层。

对齐主流小模型（Qwen / Llama 系）的成熟选择：
  · GQA：减少 KV 头、显著降低 KV 缓存与内存开销，天然支持长上下文
  · RoPE：旋转位置编码，外推性好，支持超出训练长度的位置
  · 无绝对位置嵌入，位置信息完全由 RoPE 注入
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "precompute_cos_sin", "rotate_half", "apply_rotary"]


def precompute_cos_sin(head_dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 的 cos/sin 缓存。

    Returns:
        cos: (max_seq_len, head_dim)
        sin: (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim 必须为偶数以支持 RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)                  # (L, head_dim/2)
    angles = torch.cat((angles, angles), dim=-1)       # (L, head_dim)，两半同频
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rotate_half(x):
    """把最后维两半对调并取反（标准 RoPE 旋转实现）。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    """应用旋转位置编码。

    Args:
        q/k: (B, H, T, head_dim)
        cos/sin: (T, head_dim) 或 (1, 1, T, head_dim)
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class Attention(nn.Module):
    """GQA + RoPE 自注意力。

    Args:
        d_model:     模型宽度
        n_head:      Query 头数
        n_kv_head:   KV 头数（GQA 分组数，须整除 n_head）
        head_dim:    每头维度
        max_seq_len: 预计算 RoPE 缓存的最大长度
        dropout:     注意力 dropout（训练用）
        bias:        是否使用偏置
        theta:       RoPE 频率基
    """

    def __init__(self, d_model, n_head, n_kv_head, head_dim, max_seq_len,
                 dropout=0.0, bias=False, theta=10000.0):
        super().__init__()
        assert d_model == n_head * head_dim, "d_model 必须等于 n_head * head_dim"
        assert n_head % n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.d_model = d_model
        self.repeats = n_head // n_kv_head
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, (n_head + 2 * n_kv_head) * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_head * head_dim, d_model, bias=bias)

        cos, sin = precompute_cos_sin(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x)                                     # (B, T, (H+2KV)*hd)
        q, k, v = torch.split(
            qkv,
            [self.n_head * self.head_dim,
             self.n_kv_head * self.head_dim,
             self.n_kv_head * self.head_dim],
            dim=-1,
        )
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B,H,T,hd)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B,KV,T,hd)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE（仅对 q、k 注入位置）
        q, k = apply_rotary(q, k, self.cos[:T], self.sin[:T])

        # GQA：KV 头按需复制到 query 头数
        k = k.repeat_interleave(self.repeats, dim=1)          # (B,H,T,hd)
        v = v.repeat_interleave(self.repeats, dim=1)

        att = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)   # (B,H,T,T)
        # 因果掩码
        causal = torch.triu(torch.full((T, T), float("-inf"),
                                       device=x.device), diagonal=1)
        att = att + causal
        att = F.softmax(att, dim=-1)
        if self.dropout > 0 and self.training:
            att = F.dropout(att, p=self.dropout)
        y = att @ v                                               # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.out_proj(y)                                   # (B,T,d_model)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Attention(d_model=64, n_head=4, n_kv_head=1, head_dim=16, max_seq_len=128)
    x = torch.randn(2, 32, 64)
    y = m(x)
    y.mean().backward()
    print("Attention OK:", tuple(y.shape))
