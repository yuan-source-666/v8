"""整体 CausalLM：embedding → 主骨干 → lm_head（可选权重绑定）。

初始化策略（对齐架构文档 §1.2）：
  · 去偏置：所有线性层 bias=False（模型不引入任何可学习偏置）
  · 深度缩放（scaled init）：残差分支的最后一层投影
    （Attention.out_proj / Mamba.out_proj / FFN.down）以 std/√(2n_layer) 初始化，
    保证深网络残差前向增益稳定；其余投影 std=0.02，
    lm_head（未绑定时）按 1/√d_model 初始化。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid import RMSNorm, HybridBackbone

__all__ = ["MambaMixGPT", "build_model", "load_model_config"]


class MambaMixGPT(nn.Module):
    """Hybrid Mamba-Transformer 因果语言模型。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg["vocab_size"])
        self.d_model = int(cfg["d_model"])
        self.n_layer = int(cfg["n_layer"])
        self.max_seq_len = int(cfg.get("max_seq_len", 512))
        tie = bool(cfg.get("tie_embeddings", True))

        # —— embedding ——
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        # —— 混合骨干 ——
        mcfg = dict(cfg.get("mamba", {}))
        attn_cfg = dict(
            n_head=int(cfg["n_head"]),
            n_kv_head=int(cfg["n_kv_head"]),
            head_dim=int(cfg.get("head_dim", 32)),
            max_seq_len=self.max_seq_len,
        )
        self.backbone = HybridBackbone(
            d_model=self.d_model,
            n_layer=self.n_layer,
            layout_str=str(cfg.get("layout", "MMA")),
            ffn_mult=float(cfg.get("ffn_mult", 4)),
            mamba_cfg=mcfg,
            attn_cfg=attn_cfg,
        )
        self.norm_f = RMSNorm(self.d_model)
        # —— 输出头（可选权重绑定）——
        self.tie_embeddings = tie
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------ #
    # 去偏置 + 深度缩放初始化
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        std = 0.02
        depth = math.sqrt(2 * self.n_layer)      # scaled init 分母
        out_std = std / depth                    # 残差末层投影

        def norm_(t, s):
            nn.init.normal_(t, mean=0.0, std=s)

        norm_(self.tok_emb.weight, std)
        if not self.tie_embeddings:
            norm_(self.lm_head.weight, 1.0 / math.sqrt(self.d_model))

        for blk in self.backbone.blocks:
            # —— 子层（Mamba2 / Attention）——
            sub = blk.sublayer
            name = type(sub).__name__
            if name == "Attention":
                norm_(sub.qkv.weight, std)
                norm_(sub.out_proj.weight, out_std)
            elif name == "Mamba2":
                norm_(sub.in_proj.weight, std)
                norm_(sub.x_proj.weight, std)
                norm_(sub.conv1d.weight, std)
                norm_(sub.out_proj.weight, out_std)
            # —— SwiGLU 前馈：gate/up 常规，down 深度缩放 ——
            ffn = blk.ffn
            norm_(ffn.gate.weight, std)
            norm_(ffn.up.weight, std)
            norm_(ffn.down.weight, out_std)
        return self

    # ------------------------------------------------------------------ #
    # 前向：logits 与（可选）交叉熵损失
    # ------------------------------------------------------------------ #
    def forward(self, idx, targets=None, autocast_dtype=None):
        """idx: (B, T) LongTensor；targets: (B, T) LongTensor（-100 表示忽略）。"""
        B, T = idx.shape
        assert T <= self.max_seq_len, f"序列长度 {T} 超过 max_seq_len={self.max_seq_len}"
        x = self.tok_emb(idx)                    # (B, T, d_model)
        x = self.backbone(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)                 # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    # ------------------------------------------------------------------ #
    # 学习率调度：warmup + cosine（供 train.py 复用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def lr_at(step, max_iters, lr, min_lr, warmup_iters):
        """step 从 0 开始；warmup 内线性升，其后 cosine 衰减到 min_lr。"""
        if warmup_iters > 0 and step < warmup_iters:
            frac = (step + 1) / warmup_iters
            return lr * frac
        if step >= max_iters:
            return min_lr
        progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (lr - min_lr)

    # ------------------------------------------------------------------ #
    # 优化器分组：norm/无显式参数 → 权重衰减 0（还原 embedding 正常衰减）
    # ------------------------------------------------------------------ #
    def configure_optimizers(self, weight_decay=0.1, betas=(0.9, 0.95), lr=1e-3):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:                       # 权重矩阵 → 衰减
                decay.append(p)
            else:                                 # norm / bias / 标量参数 → 不衰减
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------- #
# 配置加载 + 模型构建
# ---------------------------------------------------------------------- #
def resolve_vocab_size(cfg):
    """词表大小以 tokenizer 的真实词表为准，避免配置写死过小导致 embedding 越界。

    tiktoken 的 gpt2 词表为 50257（含 <|endoftext|>）；当配置同时指定
    tokenizer=gpt2 时，运行时解析实际词表并覆盖配置里的 vocab_size。
    """
    tok = str(cfg.get("tokenizer", "gpt2")).lower()
    if tok in ("gpt2", "gpt2_bpe"):
        import tiktoken
        return tiktoken.get_encoding("gpt2").n_vocab
    return int(cfg.get("vocab_size", 50257))


def load_model_config(config_path, profile):
    """读取 YAML 配置中指定档位（profile）的模型超参。

    支持顶层 `common:` 默认值与各档（mini/tiny）的覆盖（yaml anchor 合并）。
    """
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profile_cfg = raw.get(profile)
    if profile_cfg is None:
        raise ValueError(f"配置 {config_path} 中不存在档位: {profile!r}（可选: {list(raw.keys())}）")
    common = dict(raw.get("common", {}))
    merged = dict(common)
    for k, v in profile_cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}        # 深层 merge（如 mamba）
        else:
            merged[k] = v
    return merged


def build_model(config_path, profile):
    """从 YAML 配置构建 MambaMixGPT 混合模型。"""
    cfg = load_model_config(config_path, profile)
    model_cfg = dict(cfg)
    model_cfg["vocab_size"] = resolve_vocab_size(cfg)   # 以 tokenizer 实际词表为准
    model = MambaMixGPT(model_cfg)
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/mini.yaml")
    p.add_argument("--profile", default="mini")
    args = p.parse_args()
    torch.manual_seed(0)
    m = build_model(args.config, args.profile)
    x = torch.randint(0, m.vocab_size, (2, 128))
    logits, loss = m(x, x)
    loss.backward()
    print(f"build OK: logits={tuple(logits.shape)} loss={loss.item():.4f} "
          f"params(M)={m.num_params() / 1e6:.3f} type-of-blocks={''.join(m.backbone.types)}")
