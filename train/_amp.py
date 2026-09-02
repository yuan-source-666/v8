"""训练环境兼容层：bf16 反向自动降级保护。

背景（实测，Intel Ultra 7 255H / CPU-only，torch 2.13.0+cpu）：
  oneDNN(DNNL) 在本平台（avx2_vnni_2，无 bf16 反向内核）不支持 bf16 反向，
  forward 正常、backward 抛 RuntimeError("DNNL does not support bf16/f16 backward...")。
  关闭 torch.backends.mkldnn 后，原生 PyTorch bf16 前向+反向可正常工作。

ensure_amp_safe()：对一个小网络做一次 bf16 forward+backward 探测，
  命中 DNNL 不支持 → 自动关闭 oneDNN 并返回模式标志；随后训练脚本即可安全使用 bf16。
"""
from __future__ import annotations

import torch


def model_amp_probe(model, autocast_ctx) -> str:
    """在【真实模型】上做一次 bf16 前向+反向探测，返回实际可用精度。

    比 ensure_amp_safe 的小 Linear 探测更可靠：命中平台特定的算子（如
    Mamba 的 causal conv1d bf16 反向在部分指令集上不被 DNNL 支持）。
    探测前强制恢复 oneDNN，命中“DNNL does not support bf16”则关闭 oneDNN
    后再探测一次；若 bf16 仍不可用返回 float32。
    """
    torch.backends.mkldnn.enabled = True
    xp = torch.randint(0, model.vocab_size, (1, 32))
    tp = torch.randint(0, model.vocab_size, (1, 32))
    try:
        with autocast_ctx():
            _, lp = model(xp, tp)
        lp.backward()
        model.zero_grad(set_to_none=True)
        return "bf16"
    except RuntimeError as e:
        if "DNNL" in str(e):
            torch.backends.mkldnn.enabled = False
            try:
                xq = torch.randint(0, model.vocab_size, (1, 32))
                tq = torch.randint(0, model.vocab_size, (1, 32))
                with autocast_ctx():
                    _, lq = model(xq, tq)
                lq.backward()
                model.zero_grad(set_to_none=True)
                return "bf16"
            except Exception:
                return "float32"
        return "float32"


def ensure_amp_safe(dtype: str = "bf16", verbose: bool = True) -> str:
    """返回实际可用的精度。dtype=bf16 且平台不支持时自动关 oneDNN 并保持 bf16；
    兜底若 bf16 仍无法运行则返回 float32。"""
    if dtype != "bf16":
        return dtype
    probe = torch.nn.Linear(16, 8)
    xp = torch.randn(4, 16, dtype=torch.bfloat16)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yp = probe(xp).sum()
        yp.backward()
        return "bf16"
    except RuntimeError as e:
        if "DNNL" in str(e):
            torch.backends.mkldnn.enabled = False
            if verbose:
                print("[v8/amp] 平台 oneDNN 不支持 bf16 反向，已自动关闭 oneDNN（保留 bf16 原生训练）")
            # 关掉 oneDNN 后再探测一次
            try:
                probe2 = torch.nn.Linear(16, 8)
                y2 = probe2(xp).sum()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    y2 = probe2(xp).sum()
                y2.backward()
                return "bf16"
            except RuntimeError as e2:
                if verbose:
                    print(f"[v8/amp] bf16 仍不可用（{e2}），降级为 float32")
                return "float32"
        else:
            if verbose:
                print(f"[v8/amp] bf16 探测异常（{e}），降级为 float32")
            return "float32"
