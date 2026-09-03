"""预训练主循环（CPU-only 适配）。

对齐架构文档 §2.2 关键工程决策与 §1.6 硬件适配：
  · IPEX / oneDNN 加速（IPEX 缺失时自动退回原生 oneDNN，不影响正确性）
  · torch.set_num_threads(16)，充分利用 16 核 16 线程
  · bf16 CPU 混合精度（autocast）
  · 梯度累积（grad_accum）以放大有效 batch
  · Warmup + Cosine 学习率调度 + 梯度裁剪
  · checkpoint 保存 / 续训（--resume）
  · 可开关的 drives 内在奖励（--use-drives）：
      - 驱动损失附加到总损失（日志可见）
      - 恐惧越界时降低有效学习率（稳态刹车，真实影响优化）
      - 记录 desire/fear/deviation 指标
  · 日志：step / loss / ppl / lr / tokens·s⁻¹ / drives 指标
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model, load_model_config, MambaMixGPT
from _amp import model_amp_probe
from _drives import DrivesController

# ---------------- 环境加速（线程 / oneDNN / IPEX）----------------
def setup_runtime(cfg, verbose=True):
    nth = int(cfg.get("num_threads", 16))
    torch.set_num_threads(nth)
    torch.backends.mkldnn.enabled = True
    torch.manual_seed(int(cfg.get("seed", 42)))
    if verbose:
        print(f"[v8/train] CPU threads={torch.get_num_threads()} "
              f"mkldnn={torch.backends.mkldnn.enabled} "
              f"total_cpus={os.cpu_count()}")

    has_ipex = False
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        has_ipex = True
    except Exception:
        has_ipex = False
    if verbose:
        print(f"[v8/train] IPEX={'可用（启用加速）' if has_ipex else '不可用（退回原生 oneDNN）'}")
    return has_ipex


# ---------------- 数据读取（mmap，训练循环内禁止实时分词） ----------------
class DataLoader:
    def __init__(self, path, block_size, batch_size, seed=0):
        self.mem = np.load(path, mmap_mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.len = int(len(self.mem))
        self.gen = torch.Generator().manual_seed(seed)

    def get_batch(self, device="cpu"):
        hi = self.len - self.block_size - 1
        if hi < 1:
            raise RuntimeError("语料太短，无法切出序列。请加大语料或减小 max_seq_len。")
        ix = torch.randint(0, hi, (self.batch_size,), generator=self.gen)
        xs, ys = [], []
        for i in ix.tolist():
            xs.append(torch.from_numpy(self.mem[i:i + self.block_size].astype(np.int64)))
            ys.append(torch.from_numpy(self.mem[i + 1:i + 1 + self.block_size].astype(np.int64)))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


# ---------------- 驱动信号层集成 ----------------
# 统一走 _drives.DrivesController（与 sft.py / dpo.py 共用同一实现），
# 旧行内版 make_drives 已移除 —— 见 _drives.py 模块文档。


# ---------------- 评估（验证集损失） ----------------
@torch.no_grad()
def evaluate(model, loader, num_batches, autocast_ctx):
    # 语料过短（切不出完整序列）时跳过 eval，避免 demo/极小语料下结尾崩溃
    if loader.len < loader.block_size + 2:
        print(f"[v8/train] ⚠️ 验证语料过短（{loader.len} tokens < block_size+2={loader.block_size + 2}），跳过本次 eval")
        return float("inf")
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = loader.get_batch()
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="v8 预训练")
    ap.add_argument("--config", default="config/mini.yaml")
    ap.add_argument("--profile", default="mini")
    ap.add_argument("--resume", default=None, help="续训 checkpoint 路径")
    ap.add_argument("--use-drives", action="store_true", help="启用驱动信号层内在奖励")
    ap.add_argument("--max-iters", type=int, default=None, help="覆盖配置的最大步数")
    ap.add_argument("--dtype", default=None, choices=["bf16", "float32"])
    ap.add_argument("--out", default=None, help="覆盖输出目录")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_model_config(args.config, args.profile)
    if args.seed is not None:
        cfg["seed"] = args.seed
    use_drives = args.use_drives or bool(cfg.get("use_drives", False))
    has_ipex = setup_runtime(cfg)

    out_dir = args.out or os.path.join(cfg.get("out_dir", "out"), cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    # —— 目标精度（bf16 混合精度 / float32）——
    dtype = args.dtype or cfg.get("dtype", "bf16")
    autocast_ctx = (lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)
                    if dtype == "bf16" else _NullCtx())

    # —— 模型 ——
    model = build_model(args.config, args.profile)
    # —— 真机一次性探测：bf16 反向不被 oneDNN 支持时自动关闭 oneDNN（保留 bf16）——
    dtype = model_amp_probe(model, autocast_ctx)
    if dtype == "bf16":
        print(f"[v8/train] 混合精度：bf16 (autocast, CPU; mkldnn={torch.backends.mkldnn.enabled})")
    else:
        autocast_ctx = lambda: _NullCtx()
        print("[v8/train] 精度：float32（平台不支持 bf16 反向，已自动降级）")
    if has_ipex:
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model)
            print("[v8/train] 已应用 ipex.optimize")
        except Exception as e:
            print(f"[v8/train] ipex.optimize 失败，继续原生运行: {e}")
    print(f"[v8/train] 模型={cfg['name']} 参数量(M)={model.num_params()/1e6:.3f} "
          f"层={cfg['n_layer']} d={cfg['d_model']} 布局={''.join(model.backbone.types)}")

    # —— 数据 ——
    block_size = int(cfg["max_seq_len"])
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg.get("grad_accum", 8))
    train_path = cfg["train_data"]
    val_path = cfg["val_data"]
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[v8/train] 未找到 token 缓存，请先运行： python data/prepare.py "
              f"--config {args.config} --profile {args.profile} （或加 --demo）")
        sys.exit(1)
    train_loader = DataLoader(train_path, block_size, batch_size)
    val_loader = DataLoader(val_path, block_size, batch_size, seed=999)

    # —— 优化器 ——
    lr = float(cfg["lr"])
    max_iters = int(args.max_iters or cfg.get("max_iters", 2000))
    warmup_iters = int(cfg.get("warmup_frac", 0.02) * max_iters)
    min_lr = lr * float(cfg.get("min_lr_frac", 0.1))
    optimizer = model.configure_optimizers(weight_decay=float(cfg.get("weight_decay", 0.1)), lr=lr)
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # —— 驱动信号层 ——
    ctrl = DrivesController(cfg) if use_drives else None
    if ctrl is not None:
        print("[v8/train] drives 已启用（L1 可微稳态正则 + L2 恐惧刹车/内在奖励）")

    # —— 续训 / 断点 ——
    step = 0
    best_val = float("inf")
    model_path_best = os.path.join(out_dir, "best.pt")
    ckpt_path_last = os.path.join(out_dir, "last.pt")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"[v8/train] 已续训自 {args.resume}（step={step}）")

    def save_ckpt(path, only_model=False):
        torch.save({"model": model.state_dict()}, path) if only_model else \
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "best_val": best_val, "config": cfg,
                        "profile": args.profile}, path)

    # —— 训练循环 ——
    tokens_per_update = batch_size * block_size * grad_accum
    log_interval = int(cfg.get("log_interval", 20))
    eval_interval = int(cfg.get("eval_interval", 500))
    save_interval = int(cfg.get("save_interval", 2000))
    eval_iters = int(cfg.get("eval_iters", 60))
    running_loss = 0.0
    t_start = time.time()
    t_window = time.time()
    print(f"[v8/train] 开始训练：max_iters={max_iters} lr={lr}→{min_lr} "
          f"batch={batch_size} grad_accum={grad_accum} 每更新有效tokens={tokens_per_update}")

    for step in range(step, max_iters):
        # — 学习率（含 drives 恐惧刹车：用上一步 fear，趋势逼近边界即降 LR）——
        lr_now = MambaMixGPT.lr_at(step, max_iters, lr, min_lr, warmup_iters)
        brake = ctrl.brake() if ctrl is not None else 1.0
        for g in optimizer.param_groups:
            g["lr"] = lr_now * brake

        # — 梯度累积微批（含 L1 可微稳态正则）—
        model.train()
        micro_losses = []
        rms_vals = []
        for _ in range(grad_accum):
            x, y = train_loader.get_batch()
            with autocast_ctx():
                logits, loss = model(x, y)
            ce = loss.float().item()
            if ctrl is not None:
                stab, rms = ctrl.stability_loss(logits)   # 可微，梯度经 RMS 回流
                if rms is not None:
                    rms_vals.append(rms)
                loss = loss + stab
            (loss / grad_accum).backward()
            micro_losses.append(ce)

        # — 累积日志 + drives 状态推进（L2 观察式：指标 → S(t) → 恐惧/奖励）—
        mb_loss = sum(micro_losses) / len(micro_losses)
        running_loss = running_loss * 0.9 + mb_loss * 0.1
        info = None
        if ctrl is not None:
            rms_mean = sum(rms_vals) / len(rms_vals) if rms_vals else None
            info = ctrl.update(mb_loss, ctrl.grad_norm(model), act_rms=rms_mean)

        # — 参数更新 —
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # — 日志 —
        if (step + 1) % log_interval == 0:
            dt = time.time() - t_window
            tsp = tokens_per_update / max(dt, 1e-9)
            ppl = math.exp(min(mb_loss, 20))
            msg = (f"step {step + 1}/{max_iters} loss {mb_loss:.4f} (EMA {running_loss:.4f}) "
                   f"ppl {ppl:.2f} lr {lr_now * brake:.2e} {tsp / 1000:.1f}k tok/s")
            if ctrl is not None and info is not None:
                msg += ctrl.log_suffix(info)
            print(f"[v8/train] {msg}")
            t_window = time.time()

        # — 验证 + checkpoint —
        if (step + 1) % eval_interval == 0 or (step + 1) == max_iters:
            val_loss = evaluate(model, val_loader, eval_iters, autocast_ctx)
            if math.isinf(val_loss):
                print(f"[v8/train] ===== val_loss 跳过（验证语料过短，无法切出序列）=====")
            else:
                print(f"[v8/train] ===== val_loss {val_loss:.4f} ppl {math.exp(min(val_loss, 20)):.2f} =====")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(model_path_best, only_model=True)
                    print(f"[v8/train] 新最佳模型已保存 → {model_path_best}")
        if (step + 1) % save_interval == 0:
            save_ckpt(ckpt_path_last)
            print(f"[v8/train] 检查点已保存 → {ckpt_path_last}")

    # — 收尾 —
    save_ckpt(ckpt_path_last)
    save_ckpt(model_path_best, only_model=True)
    total_time = time.time() - t_start
    print(f"[v8/train] 训练完成，耗时 {total_time / 60:.1f} 分钟。"
          f"最终 checkpoint → {ckpt_path_last}、{model_path_best}")


class _NullCtx:
    """float32 模式下的空上下文管理器。"""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
