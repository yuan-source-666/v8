"""最小可运行 SFT —— 指令监督微调（自带一小份内置示例指令集）。

对齐架构文档 §2.1 Stage 2：
  · 在预训练底座（load 自 --init checkpoint）之上做指令微调
  · 采用 ChatGPT 式损失：只对“回答(output)”部分 token 计算损失，
    instruction/分隔符 部分掩码为 -100
  · 未指定 --init 时也从随机初始化开始微调（仍可运行，仅用于链路验证）
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
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model, load_model_config
from _amp import model_amp_probe
from _drives import DrivesController

EOT = "<|endoftext|>"
SEP = "\n答："

# ---------------- 内置示例指令集（轻量演示数据） ----------------
EXAMPLES = [
    {"instruction": "1 + 1 等于多少？", "output": "等于 2。"},
    {"instruction": "2 + 3 * 4 等于多少？", "output": "等于 14。先乘后加。"},
    {"instruction": "什么是 RMSNorm？", "output": "RMSNorm 是去除均值只有缩放项的归一化，按均方根归一化输入，常用于现代小模型。"},
    {"instruction": "什么是 SwiGLU？", "output": "SwiGLU 是一种门控前馈激活，用 SiLU 门控两路线性变换的乘积，表达能力更强。"},
    {"instruction": "什么是 RoPE？", "output": "RoPE 是旋转位置编码，通过旋转矩阵把位置信息注入向量，外推性较好。"},
    {"instruction": "什么是 GQA？", "output": "GQA 是分组查询注意力，多个查询头共享少数键值头，显著降低 KV 缓存和内存开销。"},
    {"instruction": "什么是 Mamba-2？", "output": "Mamba-2 是第二代状态空间模型，用 SSD 状态空间双对偶统一了 SSM 与注意力，线性复杂度。"},
    {"instruction": "太阳从哪边升起？", "output": "从东方升起。"},
    {"instruction": "一年有多少个月？", "output": "一年有 12 个月。"},
    {"instruction": "中国的首都是哪里？", "output": "北京。"},
    {"instruction": "10 的平方是多少？", "output": "100。"},
    {"instruction": "简述自我进化闭环。", "output": "自我进化闭环由数据获取、数据筛选、模型优化与推理细化四个环节组成，自动评估贯穿全程，让模型借助自身数据持续变强。"},
]


def build_dataset(enc, max_len):
    """把示例指令集编码成 (input_ids, targets) 列表；targets 仅回答部分非 -100。"""
    samples = []
    for ex in EXAMPLES:
        inst_ids = enc.encode(ex["instruction"])
        sep_ids = enc.encode(SEP)
        out_ids = enc.encode(ex["output"]) + enc.encode(EOT, allowed_special={EOT})
        full = inst_ids + sep_ids + out_ids
        if len(full) > max_len:                # 截断
            full = full[:max_len]
        input_ids = torch.tensor(full[:-1], dtype=torch.long)
        # targets 对齐：input_ids[i] 预测 full[i+1]，回答区（sep 之后）才计入损失
        n_ignore = len(inst_ids) + len(sep_ids)
        targets = [-100] * max(0, n_ignore - 1) + full[n_ignore:]
        # 上面 targets 长度 = (n_ignore-1) + (len(full)-n_ignore) = len(full)-1 = len(input_ids) ✓
        targets = torch.tensor(targets, dtype=torch.long)
        samples.append((input_ids, targets))
    return samples


def get_batch(samples, max_len, device="cpu"):
    ids, tgt = zip(*samples)
    x = torch.stack([torch.nn.functional.pad(i, (0, max_len - i.numel()), value=0) for i in ids])
    y = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.numel()), value=-100) for t in tgt])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser(description="v8 SFT 指令微调")
    ap.add_argument("--config", default="config/mini.yaml")
    ap.add_argument("--profile", default="mini")
    ap.add_argument("--init", default=None, help="预训练 checkpoint（model-only 或完整 last.pt）")
    ap.add_argument("--out", default=None, help="输出目录（默认 out/<profile>）")
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float32"])
    ap.add_argument("--use-drives", action="store_true", help="启用驱动信号层（恐惧刹车/内在奖励）")
    args = ap.parse_args()

    cfg = load_model_config(args.config, args.profile)
    torch.manual_seed(int(cfg.get("seed", 42)))
    torch.set_num_threads(int(cfg.get("num_threads", 16)))

    enc = tiktoken.get_encoding(cfg.get("tokenizer", "gpt2"))
    device = "cpu"
    model = build_model(args.config, args.profile)
    autocast_ctx = (lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)) \
        if args.dtype == "bf16" else (lambda: nullcontext())
    dtype = model_amp_probe(model, autocast_ctx)
    if dtype != "bf16":
        autocast_ctx = lambda: nullcontext()
        print("[v8/sft] 平台不支持 bf16 反向，已自动降级 float32")
    model.train()
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[v8/sft] 已加载初始权重: {args.init}")
    else:
        print("[v8/sft] 未指定 --init，从随机初始化开始（仅链路验证）")

    samples = build_dataset(enc, args.max_len)
    print(f"[v8/sft] 示例指令数={len(samples)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # —— 驱动信号层（可选）——
    ctrl = DrivesController(cfg) if args.use_drives else None
    if ctrl is not None:
        print("[v8/sft] drives 已启用（L1 可微稳态正则 + L2 恐惧刹车/内在奖励）")
    out_dir = args.out or os.path.join(cfg.get("out_dir", "out"), cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "sft.pt")

    t0 = time.time()
    for step in range(args.max_iters):
        if ctrl is not None:
            opt.param_groups[0]["lr"] = args.lr * ctrl.brake()   # 恐惧刹车
        x, y = get_batch(samples, args.max_len, device)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits, loss = model(x, y)
        ce = loss.detach().float().item()
        rms = None
        if ctrl is not None:
            stab, rms = ctrl.stability_loss(logits)   # L1 可微稳态正则（梯度经 RMS 回流）
            loss = loss + stab
        loss.backward()
        opt.step()
        if ctrl is not None:
            info = ctrl.update(ce, ctrl.grad_norm(model), act_rms=rms)
        if (step + 1) % 20 == 0:
            msg = (f"[v8/sft] step {step + 1}/{args.max_iters} loss {loss.item():.4f} "
                   f"({(time.time() - t0) / 60:.1f} min)")
            if ctrl is not None:
                msg += ctrl.log_suffix(info)
            print(msg)
    torch.save({"model": model.state_dict(), "config": cfg, "profile": args.profile}, save_path)
    print(f"[v8/sft] 完成，已保存 → {save_path}")


if __name__ == "__main__":
    main()
