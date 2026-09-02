"""drives 驱动信号层共享集成 —— train.py / sft.py / dpo.py 三处共用。

从 train.py 提炼出统一的「每训练步」drives 集成逻辑，行为保持一致，
并修复原 train.py 中 loss_var 恒为 0 的问题（原实现每步新建 deque 填同一
loss 值再求方差，方差恒为 0；本模块改为累积滑动窗口，loss_var 反映真实抖动，
使 consistency 状态分量的演化更有意义）。

用法（SFT/DPO 同构，train.py 逻辑等价）：
    ctrl = DrivesController(cfg)          # 仅当启用 drives 时创建
    ...
    for step in range(max_iters):
        lr = base_lr * ctrl.brake()        # 恐惧刹车（step 开始，用上一步 fear）
        ... 前向/反向/step ...
        info = ctrl.update(loss_val, ctrl.grad_norm(model))   # step 后推进状态
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch


def make_drives(cfg: Dict):
    """从配置构建 (InternalState, DriveSignals)。逻辑与 train.py 一致。"""
    from drives.state import InternalState, parse_specs
    from drives.signal import DriveSignals
    drv = cfg.get("drives") or {}
    specs = parse_specs(drv.get("components", {}))
    state = InternalState(specs)
    sig = DriveSignals(state, drv,
                       safety_bound=drv.get("safety_bound"),
                       target_state=drv.get("target_state"))
    return state, sig


class DrivesController:
    """封装 drives 的逐训练步集成：指标更新 → 状态推进 → 刹车 / 损失 / 奖励。"""

    def __init__(self, cfg: Dict, energy_cost: float = 0.005):
        self.cfg = cfg
        self.energy_cost = energy_cost
        self.state, self.sig = make_drives(cfg)
        self.metrics: Dict[str, float] = {
            "loss_ema": None, "loss_delta": 0.0, "loss_var": 0.0, "grad_norm": 0.0,
        }
        self._loss_window = deque(maxlen=20)   # 累积滑动窗口（修复 loss_var 恒 0）
        self.fear = 0.0                        # 上一步的恐惧（用于 brake）

    # ------------------------------------------------------------------
    def brake(self) -> float:
        """当前恐惧对应的学习率刹车系数：1 − 0.2·fear（与 train.py 一致）。"""
        return 1.0 - 0.2 * self.fear

    def grad_norm(self, model: torch.nn.Module) -> float:
        """计算当前梯度全量 L2 范数（backward 后调用）。"""
        ps = [p.grad.detach().float().norm()
              for p in model.parameters() if p.grad is not None]
        return float(sum(p * p for p in ps) ** 0.5) if ps else 0.0

    def update(self, loss_val: float, grad_norm: Optional[float] = None) -> Dict:
        """在一个训练更新周期结束后调用：更新指标 → 推进状态 → 返回驱动信息。

        Args:
            loss_val: 本周期平均 loss（float）。
            grad_norm: 本周期梯度范数；不传则沿用上一值。
        Returns:
            dict：fear / brake / drv_loss / reward / deviation / desire / critical / state
        """
        from drives.rewards import drives_loss, intrinsic_reward, compute_state_deltas
        drop = float(loss_val)
        ema = self.metrics["loss_ema"]
        self.metrics["loss_ema"] = drop if ema is None else 0.9 * ema + 0.1 * drop
        self.metrics["loss_delta"] = drop - self.metrics["loss_ema"]
        self._loss_window.append(drop)
        self.metrics["loss_var"] = (
            float(np.var(list(self._loss_window))) if len(self._loss_window) >= 2 else 0.0)
        if grad_norm is not None:
            self.metrics["grad_norm"] = float(grad_norm)
        state_metrics = dict(self.metrics)
        state_metrics["loss"] = drop
        deltas = compute_state_deltas(self.state, self.sig, state_metrics,
                                      energy_cost=self.energy_cost)
        self.state.step(deltas)
        extra = float(drives_loss(self.sig, self.cfg).item())
        internal_reward = intrinsic_reward(self.sig, self.cfg)
        self.fear = float(self.sig.fear())
        return {
            "fear": self.fear,
            "brake": self.brake(),
            "drv_loss": extra,
            "reward": internal_reward,
            "deviation": float(self.sig.total_deviation()),
            "desire": float(self.sig.desire_total()),
            "critical": bool(self.sig.safety_critical()),
            "state": self.state.snapshot(),
        }

    def log_suffix(self, info: Dict) -> str:
        """把驱动信息格式化为日志尾缀（train.py 日志风格）。"""
        s = info["state"]
        return (f" | D {info['deviation']:.3f} desire {info['desire']:.3f} "
                f"fear {info['fear']:.3f} drv_loss {info['drv_loss']:.4f} "
                f"r {info['reward']:+.3f} E {s.get('energy', float('nan')):.2f} "
                f"R {s.get('resources', float('nan')):.2f} "
                f"C {s.get('consistency', float('nan')):.2f} "
                f"M {s.get('safety_margin', float('nan')):.2f} "
                f"[{'临界' if info['critical'] else '稳态'}]")
