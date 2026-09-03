"""drives 驱动信号层共享集成 —— train.py / sft.py / dpo.py 三处共用。

从 train.py 提炼出统一的「每训练步」drives 集成逻辑，行为保持一致，
并修复原 train.py 中 loss_var 恒为 0 的问题（原实现每步新建 deque 填同一
loss 值再求方差，方差恒为 0；本模块改为累积滑动窗口，loss_var 反映真实抖动，
使 consistency 状态分量的演化更有意义）。

两级接入（让 drives 真正“活”在训练里）：
  · L1 张量级（可微）：stability_loss(logits) 把 logits 的 RMS 拉向历史稳态
    基线（EMA），权重由驱动信号动态调制（恐惧/偏差越大约束越强），
    梯度经 RMS 流回网络 —— 驱动层由此真实参与优化目标。
  · L2 状态级（观察式）：update() 推进 S(t)，fear 通过 brake() 调低有效
    学习率（趋势逼近安全边界时“踩刹车”）。

用法（SFT/DPO 同构，train.py 逻辑等价）：
    ctrl = DrivesController(cfg)          # 仅当启用 drives 时创建
    ...
    for step in range(max_iters):
        lr = base_lr * ctrl.brake()        # 恐惧刹车（step 开始，用上一步 fear）
        logits, loss = model(x, y)
        stab, rms = ctrl.stability_loss(logits)   # 可微稳态正则
        (loss + stab).backward()
        ... opt.step() ...
        info = ctrl.update(loss_val, ctrl.grad_norm(model), act_rms=rms)
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional, Tuple

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
    """封装 drives 的逐训练步集成：指标更新 → 状态推进 → 刹车 / 稳态正则 / 奖励。"""

    def __init__(self, cfg: Dict, energy_cost: float = 0.005):
        self.cfg = cfg
        drv = cfg.get("drives") or {}
        self.energy_cost = energy_cost
        self.energy_recovery = float(drv.get("energy_recovery", 0.004))
        self.w_stab = float(drv.get("w_stab", 0.05))       # L1 稳态正则基础权重
        self.act_ema_beta = float(drv.get("act_ema_beta", 0.9))
        self.state, self.sig = make_drives(cfg)
        self.metrics: Dict[str, float] = {
            "loss_ema": None, "loss_delta": 0.0, "loss_var": 0.0, "grad_norm": 0.0,
        }
        self._loss_window = deque(maxlen=20)   # 累积滑动窗口（修复 loss_var 恒 0）
        self._act_ema: Optional[float] = None  # logits RMS 稳态基线（EMA）
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

    def stability_weight(self) -> float:
        """驱动信号调制的稳态正则权重：恐惧/偏差越大，稳态约束越强。"""
        return self.w_stab * (1.0 + float(self.sig.fear())
                              + float(self.sig.total_deviation()))

    def stability_loss(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[float]]:
        """L1 张量级稳态正则（可微）：logits RMS 拉向历史稳态基线。

        首次调用只记录基线（EMA 初始化），返回 0 损失；此后每步以
        _act_ema 为 setpoint，产生真实的梯度通路。
        Returns:
            (loss, rms)：可微 0 维张量 + detached RMS 标量（供 update 跟踪）。
        """
        from drives.rewards import stability_loss
        if self._act_ema is None:
            rms = float(logits.detach().float().pow(2).mean().sqrt())
            self._act_ema = rms
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, rms
        return stability_loss(logits, self._act_ema, self.stability_weight())

    def update(self, loss_val: float, grad_norm: Optional[float] = None,
               act_rms: Optional[float] = None) -> Dict:
        """在一个训练更新周期结束后调用：更新指标 → 推进状态 → 返回驱动信息。

        Args:
            loss_val: 本周期平均 loss（float）。
            grad_norm: 本周期梯度范数；不传则沿用上一值。
            act_rms: 本周期 logits RMS（detached）；用于更新稳态基线与 act_drift。
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
        if act_rms is not None:
            if self._act_ema is None:
                self._act_ema = float(act_rms)
            self.metrics["act_drift"] = abs(float(act_rms) - self._act_ema)
            self._act_ema = (self.act_ema_beta * self._act_ema
                             + (1.0 - self.act_ema_beta) * float(act_rms))
        state_metrics = dict(self.metrics)
        state_metrics["loss"] = drop
        deltas = compute_state_deltas(self.state, self.sig, state_metrics,
                                      energy_cost=self.energy_cost,
                                      energy_recovery=self.energy_recovery)
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
        drift = self.metrics.get("act_drift", 0.0)
        return (f" | D {info['deviation']:.3f} desire {info['desire']:.3f} "
                f"fear {info['fear']:.3f} brake {info['brake']:.3f} "
                f"drift {drift:.3f} drv_loss {info['drv_loss']:.4f} "
                f"r {info['reward']:+.3f} E {s.get('energy', float('nan')):.2f} "
                f"R {s.get('resources', float('nan')):.2f} "
                f"C {s.get('consistency', float('nan')):.2f} "
                f"M {s.get('safety_margin', float('nan')):.2f} "
                f"[{'临界' if info['critical'] else '稳态'}]")
