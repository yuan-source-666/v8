"""把驱动信号接入训练：可微稳态正则（L1 张量级）+ 内在奖励 / 驱动损失（L2 状态级）。

对齐架构文档 v1.2：驱动信号层不是摆设，而是真实接入训练循环。分两级：

  · L1 张量级（可微，真实梯度通路）—— stability_loss(logits, setpoint, weight)
        把网络输出 logits 的 RMS 拉向稳态设定值 setpoint（由 S(t) 状态层的
        EMA 基线给出，即“网络激活尺度不要漂移”的稳态约束）：
            L_stab = weight · (rms(logits) − setpoint)²
        梯度经 rms 流回网络权重；weight 由驱动信号动态调制
        （恐惧/偏差越大，稳态约束越强，见 DrivesController.stability_loss）。

  · L2 状态级（观察式，作用于学习率与日志）——
        intrinsic_reward(signals) / drives_loss(signals, cfg)：
          reward = −w_reg·D + w_desire·(1−desire) − w_fear·fear
        S(t) 本身是实数值标量（非可微张量），其导出信号不直接进梯度，
        而是通过“恐惧刹车”（调低有效学习率）与稳态正则权重调制影响优化。

内部状态 S(t) 的演化由训练动态决定（compute_state_deltas），
能量、资源、一致性、安全边距互相耦合，形成趋利避害的闭环。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .signal import DriveSignals

__all__ = ["stability_loss", "intrinsic_reward", "drives_loss", "compute_state_deltas"]


def stability_loss(logits: torch.Tensor, setpoint: float,
                   weight: float) -> Tuple[torch.Tensor, float]:
    """L1 张量级稳态正则（可微）：把 logits 的 RMS 拉向稳态设定值。

    Args:
        logits:   (B, T, V) 模型输出（含梯度路径）
        setpoint: 稳态设定值（标量，来自状态层的 RMS EMA 基线，不参与梯度）
        weight:   正则权重（标量，由驱动信号动态调制）
    Returns:
        (loss, rms)：loss 为 0 维可微张量；rms 为 detached 标量（供状态层跟踪基线）。
    """
    rms = logits.float().pow(2).mean(dim=-1).sqrt().mean()   # 可微标量
    loss = weight * (rms - float(setpoint)) ** 2
    return loss, float(rms.detach())


def _weights(cfg: Dict) -> Dict[str, float]:
    d = cfg.get("drives", cfg)
    return {
        "w_reg": float(d.get("w_reg", 0.01)),
        "w_desire": float(d.get("w_desire", 0.5)),
        "w_fear": float(d.get("w_fear", 1.0)),
    }


def intrinsic_reward(signals: DriveSignals, cfg: Dict) -> float:
    """计算内在奖励（数值越大越“好”）。"""
    w = _weights(cfg)
    dev = float(signals.total_deviation())
    desire = float(signals.desire_total())
    fear = float(signals.fear())
    reward = (-w["w_reg"] * dev
              + w["w_desire"] * (1.0 - desire)
              - w["w_fear"] * fear)
    return float(reward)


def drives_loss(signals: DriveSignals, cfg: Dict, device="cpu") -> torch.Tensor:
    """附加到训练损失的驱动正则项（返回 0 维张量）。"""
    w = _weights(cfg)
    D2 = sum(d * d for d in signals.deviation().values()) / max(1, len(signals.deviation()))
    fear = float(signals.fear())
    desire = float(signals.desire_total())
    value = (w["w_reg"] * D2
             + w["w_fear"] * fear
             - w["w_desire"] * (1.0 - desire))
    return torch.tensor(value, dtype=torch.float32, device=device)


def compute_state_deltas(state, signals: DriveSignals, metrics: Dict,
                         energy_cost: float = 0.005,
                         energy_recovery: float = 0.004) -> Dict[str, float]:
    """根据训练动态把指标映射为内部状态的更新量（ΔS）。

    这是状态演化 S(t+1) = S(t) + Δ 的 Δ 来源：
      · energy：每步固定消耗（CPU 训练即“烧能量”）；学习顺利（resources 高）
        时部分回收 —— 否则能量单调衰减触底，恐惧将永久饱和，闭环失效
      · resources：损失下降 → 学到知识 → 资源上升（增量与 loss 残差负相关）
      · consistency：近期损失抖动越小、激活尺度漂移越小 → 一致性越高
      · safety_margin：梯度范数越大越危险 → 边距收缩
      · desire / fear：被 learning 信号推到与 computed desire/fear 一致
    """
    deltas: Dict[str, float] = {}

    def spec_for(name):
        return state.specs.get(name)

    # energy: 恒耗 + 学习顺利时部分回收（闭环：resources 高 → energy 止跌回升）
    if spec_for("energy"):
        recovery = 0.0
        if "resources" in state.values:
            recovery = energy_recovery * max(0.0, state.get("resources") - 0.5)
        deltas["energy"] = -energy_cost + recovery

    # resources: 由 loss 下降驱动
    if spec_for("resources") and "loss" in metrics:
        dl = metrics.get("loss_delta", 0.0)          # 本步损失变化（负=下降）
        deltas["resources"] = -0.6 * dl + 0.001      # loss 降 → resources 升

    # consistency: 由损失方差（抖动）与激活尺度漂移（act_drift）反向驱动
    if spec_for("consistency") and "loss_var" in metrics:
        d = -metrics["loss_var"] * 0.5 + 0.002
        drift = metrics.get("act_drift", 0.0)
        if drift > 0:
            d -= min(0.02, drift * 0.05)   # 激活尺度漂移越大 → 一致性越低
        deltas["consistency"] = d

    # safety_margin: 梯度范数越大越危险
    if spec_for("safety_margin") and "grad_norm" in metrics:
        gn = metrics["grad_norm"]
        deltas["safety_margin"] = 0.0 if gn < 0.5 else -min(0.02, (gn - 0.5) * 0.01)

    # desire / fear：让内部状态分量趋同于 signals 算出的驱动水平
    if spec_for("desire"):
        deltas["desire"] = (signals.desire_total() - state.get("desire"))
    if spec_for("fear"):
        deltas["fear"] = (signals.fear() - state.get("fear"))

    return deltas


if __name__ == "__main__":
    from .state import InternalState, parse_specs
    specs = parse_specs({
        "resources": {"setpoint": 0.75, "low": 0.0, "high": 1.0, "step": 0.02},
        "energy": {"setpoint": 0.8, "low": 0.0, "high": 1.0, "step": 0.02},
        "consistency": {"setpoint": 0.7, "low": 0.0, "high": 1.0, "step": 0.02},
        "safety_margin": {"setpoint": 0.6, "low": 0.0, "high": 1.0, "step": 0.02},
        "desire": {"setpoint": 0.5, "low": 0.0, "high": 1.0, "step": 0.02},
        "fear": {"setpoint": 0.3, "low": 0.0, "high": 1.0, "step": 0.02},
    })
    from .signal import DriveSignals
    state = InternalState(specs)
    sig = DriveSignals(state, {})
    cfg = {"drives": {"w_reg": 0.01, "w_desire": 0.5, "w_fear": 1.0}}
    # 1) 可微稳态正则：梯度应能经 rms 流回
    logits = torch.randn(2, 8, 16, requires_grad=True)
    setpoint = float(logits.detach().float().pow(2).mean().sqrt())
    stab, rms = stability_loss(logits, setpoint, weight=0.05)
    stab.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "稳态正则无梯度!"
    print("stability_loss OK: loss=", round(stab.detach().item(), 6), "rms=", round(rms, 4),
          "grad 流通 ✓")
    # 2) 状态演化闭环：能量下滑但学习顺利（resources 高）时应止跌
    for i in range(30):
        state.step(compute_state_deltas(state, sig,
                                        {"loss": 4.0, "loss_delta": -0.3, "loss_var": 0.02,
                                         "grad_norm": 0.8, "act_drift": 0.1}))
    s = state.snapshot()
    print("rewards OK: reward=", round(intrinsic_reward(sig, cfg), 4),
          "loss=", round(float(drives_loss(sig, cfg)), 4),
          "fear=", round(sig.fear(), 4),
          "state=", {k: round(v, 3) for k, v in s.items()})
