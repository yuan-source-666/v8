"""最小可运行 DPO —— 直接偏好优化（自带枚举偏好对）。

对齐架构文档 §2.1 Stage 3（DPO 首选：无需训练奖励模型、实现简单、1B 以下彻底可行）：

  loss = −E[ log σ( β·( logπ_w/π_ref_w − logπ_l/π_ref_l ) ) ]

其中 π_ref 为冻结的参考模型（本实现从 π 深拷贝权重，eval + no_grad）。
仅对响应(response)部分 token 计算序列对数概率（prompt 掩码掉），
同一偏好对内的 chosen/rejected 响应截齐到等长，保证 loss 有效。

用法：
    python train/dpo.py --config config/mini.yaml --profile mini \
        --init out/mini/sft.pt
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model, load_model_config
from _amp import model_amp_probe

EOT = "<|endoftext|>"
SEP = "\n答："

# ---------------- 内置枚举偏好对（轻量演示数据） ----------------
PAIRS = [
    {"prompt": "1 + 1 等于多少？", "chosen": "等于 2。", "rejected": "等于 3。"},
    {"prompt": "2 + 3 * 4 等于多少？", "chosen": "等于 14。先乘后加。", "rejected": "等于 20。"},
    {"prompt": "太阳从哪边升起？", "chosen": "从东方升起。", "rejected": "从西方升起。"},
    {"prompt": "中国的首都是哪里？", "chosen": "北京。", "rejected": "上海。"},
    {"prompt": "10 的平方是多少？", "chosen": "100。", "rejected": "20。"},
    {"prompt": "什么是 RoPE？", "chosen": "旋转位置编码，通过旋转矩阵注入位置信息。", "rejected": "一种随机初始化的词嵌入。"},
    {"prompt": "什么是 GQA？", "chosen": "分组查询注意力，多个查询头共享少数 KV 头。", "rejected": "一种把查询头数量翻倍的注意力。"},
    {"prompt": "什么是一年？", "chosen": "一年通常有 12 个月或 365 天。", "rejected": "一年有 100 个月。"},
]


def make_tensor(enc, prompt, response, max_len):
    """返回 (inp_ids, resp_mask)；resp_mask=1 表示该位置属于响应(token 对齐)。

    输入序列 = prompt_tokens + sep + response_tokens + eot
    resp_mask 标记 response+eot 部分。
    """
    p = enc.encode(prompt)
    s = enc.encode(SEP)
    r = enc.encode(response) + enc.encode(EOT)
    full = p + s + r
    full = full[:max_len]
    inp = torch.tensor(full, dtype=torch.long)
    n_ignore = len(p) + len(s)
    mask = torch.zeros(len(full), dtype=torch.float32)
    mask[n_ignore:] = 1.0
    return inp, mask


def seq_logprobs(model, inp, mask, autocast_ctx):
    """计算给定 mask 位置的平均负对数似然对数概率（标量张量）。"""
    inp = inp.unsqueeze(0)
    mask = mask.unsqueeze(0)
    with torch.no_grad():
        with autocast_ctx():
            logits, _ = model(inp)                 # (1, T, V)
    logp = F.log_softmax(logits.float(), dim=-1)   # (1, T, V)
    target = torch.roll(inp, -1, dims=1)           # 预测下一 token
    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, T)
    denom = mask.sum(dim=-1).clamp(min=1)
    return (gathered * mask).sum(dim=-1) / denom   # (1,)


def main():
    ap = argparse.ArgumentParser(description="v8 DPO 偏好对齐")
    ap.add_argument("--config", default="config/mini.yaml")
    ap.add_argument("--profile", default="mini")
    ap.add_argument("--init", required=False, default=None, help="SFT/预训练 checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float32"])
    args = ap.parse_args()

    cfg = load_model_config(args.config, args.profile)
    torch.manual_seed(int(cfg.get("seed", 42)))
    torch.set_num_threads(int(cfg.get("num_threads", 16)))
    enc = tiktoken.get_encoding(cfg.get("tokenizer", "gpt2"))
    device = "cpu"

    model = build_model(args.config, args.profile)
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[v8/dpo] 已加载初始权重: {args.init}")
    else:
        print("[v8/dpo] 警告：未指定 --init，从随机初始化开始（仅链路验证）")
    model.train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False

    autocast_ctx = (lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)) \
        if args.dtype == "bf16" else (lambda: nullcontext())
    dtype = model_amp_probe(model, autocast_ctx)
    if dtype != "bf16":
        autocast_ctx = lambda: nullcontext()
        print("[v8/dpo] 平台不支持 bf16 反向，已自动降级 float32")

    # 预处理偏好对（截到等长响应）
    data = []
    for ex in PAIRS:
        iw, mw = make_tensor(enc, ex["prompt"], ex["chosen"], args.max_len)
        il, ml = make_tensor(enc, ex["prompt"], ex["rejected"], args.max_len)
        min_len = min(iw.numel(), il.numel(), args.max_len)
        data.append((iw[:min_len], il[:min_len], mw[:min_len], ml[:min_len]))
    print(f"[v8/dpo] 偏好对数量={len(data)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = args.out or os.path.join(cfg.get("out_dir", "out"), cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "dpo.pt")

    t0 = time.time()
    for step in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for iw, il, mw, ml in data:
            # 参考模型 logprob（冻结）
            rw = seq_logprobs(ref, iw, mw, autocast_ctx)
            rl = seq_logprobs(ref, il, ml, autocast_ctx)
            # 策略模型 logprob（需可微 → 走 model 前向）
            with autocast_ctx():
                lw = _train_logprob(model, iw, mw)
                ll_ = _train_logprob(model, il, ml)
            ratio = args.beta * ((lw.unsqueeze(0) - rw) - (ll_.unsqueeze(0) - rl))
            loss_pair = -F.logsigmoid(ratio).mean()
            loss_pair.backward()
            total += loss_pair.item()
        opt.step()
        if (step + 1) % 10 == 0:
            print(f"[v8/dpo] step {step + 1}/{args.max_iters} loss {total / len(data):.4f} "
                  f"({(time.time() - t0) / 60:.1f} min)")
    torch.save({"model": model.state_dict(), "config": cfg, "profile": args.profile}, save_path)
    print(f"[v8/dpo] 完成，已保存 → {save_path}")


def _train_logprob(model, inp, mask):
    """可微版本：直接对策略模型返回平均对数概率（含梯度路径）。"""
    inp = inp.unsqueeze(0)
    logits, _ = model(inp)
    logp = F.log_softmax(logits.float(), dim=-1)
    target = torch.roll(inp, -1, dims=1)
    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)     # (1, T)
    maskb = mask.unsqueeze(0)
    denom = maskb.sum(dim=-1).clamp(min=1)
    return (gathered * maskb).sum(dim=-1) / denom                    # (1,) 标量


if __name__ == "__main__":
    main()
