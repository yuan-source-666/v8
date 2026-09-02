"""采样生成脚本 —— 含 self-consistency 多数投票。

用法：
    python infer/sample.py --config config/mini.yaml --profile mini \
        --init out/mini/best.pt --prompt "1 + 1 等于多少？"
    # 多数投票：同 prompt 采样 N 次，提取答案并投票
    python infer/sample.py ... --self-consistency --n-samples 5

self-consistency 说明：
  对同一 prompt 采样多个候选序列，用 --answer-regex（或默认“末段答案”提取器）
  把每个候选压缩成“答案”，再用多数投票选出最一致的答案并列出频谱。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import torch
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model, load_model_config


def top_sample(logits, temperature=1.0, top_k=0, top_p=0.0):
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_l, -1), -1)
        mask = cum - torch.softmax(sorted_l, -1) > top_p
        sorted_l[mask] = -float("inf")
        logits = sorted_l.scatter(-1, sorted_i, sorted_l)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, enc, prompt, max_new_tokens, temperature, top_k, top_p, autocast_ctx):
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        with autocast_ctx():
            for _ in range(max_new_tokens):
                logits, _ = model(x)
                nxt = top_sample(logits[0, -1], temperature, top_k, top_p)
                x = torch.cat([x, nxt], dim=1)
    out_ids = x[0, len(ids):].tolist()
    text = enc.decode(out_ids)
    # 去掉残留的 EOT
    eot = enc.encode("<|endoftext|>")[0]
    text = text.split("<|endoftext|>")[0].strip()
    return text


def default_answer(text: str) -> str:
    """默认答案提取：取最后一个非空（行）的紧凑文本。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    return lines[-1]


def main():
    ap = argparse.ArgumentParser(description="v8 采样生成")
    ap.add_argument("--config", default="config/mini.yaml")
    ap.add_argument("--profile", default="mini")
    ap.add_argument("--init", required=True, help="checkpoint 路径（model-only 或完整）")
    ap.add_argument("--prompt", default="从前有座山，")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-consistency", action="store_true",
                    help="多次采样 + 答案多数投票")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--answer-regex", default=None,
                    help="答案提取正则（第一个捕获组），覆盖默认末段提取")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float32"])
    args = ap.parse_args()

    cfg = load_model_config(args.config, args.profile)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(cfg.get("num_threads", 16)))
    enc = tiktoken.get_encoding(cfg.get("tokenizer", "gpt2"))

    model = build_model(args.config, args.profile)
    ckpt = torch.load(args.init, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"[v8/infer] 已加载 {args.init} (profile={args.profile})")

    autocast_ctx = (lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)) \
        if args.dtype == "bf16" else (lambda: torch.no_grad().__class__())

    print("=" * 70)
    print(f"Prompt: {args.prompt}")

    def do_sample(seed):
        torch.manual_seed(seed)
        return generate(model, enc, args.prompt, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p, autocast_ctx)

    if not args.self_consistency:
        text = do_sample(args.seed)
        print("-" * 70)
        print(text)
        print("=" * 70)
        return

    # —— self-consistency 多数投票 ——
    candidates = [do_sample(args.seed + i) for i in range(args.n_samples)]

    def extract(t: str) -> str:
        if args.answer_regex:
            m = re.search(args.answer_regex, t)
            return m.group(1).strip() if m else default_answer(t)
        return default_answer(t)

    answers = [extract(t) for t in candidates]
    title, cnt = Counter(answers).most_common(1)[0]
    print("-" * 70)
    for i, (t, a) in enumerate(zip(candidates, answers)):
        print(f"[候选 {i + 1}] 答案={a!r}\n{t}\n" + "-" * 70)
    print(f"[多数投票] 最一致答案 = {title!r}  (得票 {cnt}/{args.n_samples})")
    print(f"[投票频谱] {Counter(answers)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
