"""原始文本 → tokenized 缓存管线（一次性切好并落盘，训练时直接 mmap 加载）。

设计约束（对齐任务要求与架构文档 §2.2）：
  · 训练循环内**禁止**实时分词 —— 本脚本一次性完成 `文本 → token 序列`，
    并把训练/验证 token 缓存分别落盘为 .npy（memmap 可随机读）。
  · Tokenizer：tiktoken GPT-2 BPE（架构文档 Stage 0 选型）。
  · 语料来源：data/corpus/ 下所有 *.txt（UTF-8），可用 --input 指定其它目录/文件。

用法：
    python data/prepare.py --config config/mini.yaml          # 处理 data/corpus
    python data/prepare.py --config config/mini.yaml --demo   # 用内置示例语料跑通链路
    python data/prepare.py --config config/mini.yaml --force  # 重新切分（覆盖缓存）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import load_model_config

# 内置演示语料（仅用于 --demo 验证链路，非真实训练语料）
_DEMO_TEXT = """自我进化的小语言模型实验。
在一个没有独立显卡的电脑上，我们也可以用混合架构训练自己的小模型。
混合架构融合了 Mamba 状态空间模型与 Transformer 自注意力机制。
Mamba 处理长序列的效率很高，线性复杂度让它比二次复杂度的注意力更快。
注意力机制擅长检索与复制，全局视野保证了模型的信息整合能力。
两种机制以二比一的比例交替排布，组成满足需求的主骨干网络。
训练阶段包括预训练、指令微调与偏好对齐。
自我进化闭环让模型能够通过自身生成的数据持续改进。
驱动信号层使用稳态偏差机制，让欲望引导模型趋近目标，恐惧约束模型不越出安全边界。
这台机器的中央处理器是英特尔酷睿超处理器，内存三十二吉字节。
在纯中央处理器环境下，我们需要精打细算地控制数据量与模型规模。
先从最小配置开始，验证整条训练链路，再逐步扩大模型。
这是一个日拱一卒、渐进推进的实验计划。"""


def _iter_text_files(paths):
    """逐个产出 (文件名, 文本内容)，跳过空文件与非 UTF-8 可解码内容。"""
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".txt"):
            with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
                text = f.read()
            if text.strip():
                yield p, text


def prepare(args):
    cfg = load_model_config(args.config, args.profile)
    enc = tiktoken.get_encoding(cfg.get("tokenizer", "gpt2"))

    train_path, val_path = cfg["train_data"], cfg["val_data"]
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)

    # —— 1. 收集语料（真实语料 或 --demo 内置文本）——
    texts: list[str] = []
    if args.demo:
        print("[v8/prepare] 使用内置演示语料（--demo）")
        texts = [_DEMO_TEXT]
    else:
        corpus_dir = args.input_dir
        if not os.path.isdir(corpus_dir):
            print(f"[v8/prepare] 错误: 语料目录不存在 → {corpus_dir}")
            print(f"[v8/prepare] 请把 .txt 语料放进 {corpus_dir}/，或用 --demo 跑通链路。")
            raise SystemExit(1)
        files = sorted(os.path.join(corpus_dir, n) for n in os.listdir(corpus_dir))
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        if not txt_files:
            print(f"[v8/prepare] 错误: {corpus_dir} 下没有 .txt 文件。用 --demo 可跑通链路。")
            raise SystemExit(1)
        for fname, text in _iter_text_files(txt_files):
            texts.append(text)
            print(f"[v8/prepare] 载入 {len(texts)}: {fname} ({len(text)} 字符)")

    # —— 2. 一次性分词 + 拼接 ——
    raw_tokens: list[int] = []
    sample = texts[0]
    print(f"[v8/prepare] tokenizer={cfg.get('tokenizer')} 语料文件数={len(texts)} "
          f"总字符={sum(len(t) for t in texts)}")
    for i, text in enumerate(texts):
        ids = enc.encode(text)
        raw_tokens.extend(ids)
        special = enc.encode_ordinary(text)
        if args.verbose and i < 2:
            print(f"[v8/prepare]   样本{i} 前20 token: {special[:20]}")
    tokens = np.array(raw_tokens, dtype=np.uint16)
    print(f"[v8/prepare] 总 token 数: {len(tokens)}")

    # —— 3. 切分 train / val ——
    n_val = int(len(tokens) * args.val_ratio)
    n_val = max(1, min(n_val, min(4096, len(tokens) // 4)))   # val 上限保护
    train_tok, val_tok = tokens[:-n_val], tokens[-n_val:]
    print(f"[v8/prepare] val_ratio={args.val_ratio} → train={len(train_tok)} val={len(val_tok)}")

    # —— 4. 落盘（np.save; 训练端 np.load(mmap_mode='r') 随机读）——
    if not args.force and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[v8/prepare] 缓存已存在，跳过（--force 可强制重切）。")
        print(f"[v8/prepare]   train → {train_path}")
        print(f"[v8/prepare]   val   → {val_path}")
        return
    np.save(train_path, train_tok)
    np.save(val_path, val_tok)
    print(f"[v8/prepare] 完成:")
    print(f"[v8/prepare]   train → {train_path} ({len(train_tok)} tokens)")
    print(f"[v8/prepare]   val   → {val_path} ({len(val_tok)} tokens)")


def main():
    ap = argparse.ArgumentParser(description="v8 数据管线：原始文本 → token 缓存")
    ap.add_argument("--config", default="config/mini.yaml")
    ap.add_argument("--profile", default="mini", help="mini / tiny")
    ap.add_argument("--input-dir", default=None,
                    help="语料目录（默认取 config 中 data/corpus 的常见位置）")
    ap.add_argument("--demo", action="store_true", help="用内置演示语料跑通管线")
    ap.add_argument("--force", action="store_true", help="强制重新切分覆盖缓存")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.input_dir is None:
        # 默认语料目录：data/corpus（相对当前目录或脚本所在上级）
        candidates = ["data/corpus", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")]
        args.input_dir = next((c for c in candidates if os.path.isdir(c)),
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    prepare(args)


if __name__ == "__main__":
    main()
