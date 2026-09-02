"""把稳态偏差 D 接入为 intrinsic reward / loss 附加正则项（供 train.py 调用）。

对齐架构文档 v1.2：驱动信号层不是摆设，而是真实接入训练循环。
本模块提供两个可被训练直接调用的接口：

  · intrinsic_reward(signals) -> float
        RL 风格的内在奖励：
          reward = −w_reg·D       （偏离稳态 → 负向奖励，鼓励回到稳态）
                 + w_desire·desire（欲望 = 距目标距离，越近越“满足”，奖励越高）
                 − w_fear·fear    （越出安全边界的前瞻恐惧 → 强力惩罚）
        desire 项用 (1 − D_target) 语义，这里 reward 越大越“好”。

  · drives_loss(signals, cfg) -> torch.Tensor（标量张量）
        loss 附加正则项：加到 train.py 的总损失上：
          L_drive = w_reg · mean(D²) + w_fear·fear − w_desire·(1−desire)
        L_drive 随训练动态变化（因为 S(t) 由训练动态更新），从而把“稳态/趋利避害”
        作为软约束刻进优化目标。

内部状态 S(t) 本身是实数值（非可微张量），因此驱动信号通过“观察式”方式
影响训练：S(t) 的演化由训练动态决定，这里把其导出信号作为标量正则加入损失，
等价于对优化目标做动态加权。
"""
from __future__ import annotations

from typing import Dict

import torch

from .signal import DriveSignals

__all__ = ["intrinsic_reward", "drives_loss", "compute_state_deltas"]


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
                         energy_cost: float = 0.005) -> Dict[str, float]:
    """根据训练动态把指标映射为内部状态的更新量（ΔS）。

    这是状态演化 S(t+1) = S(t) + Δ 的 Δ 来源：
      · energy：每步固定消耗（CPU 训练即“烧能量”）
      · resources：损失下降 → 学到知识 → 资源上升（增量与 loss 负相关）
      · consistency：近期损失抖动越小 → 一致性越高
      · safety_margin：梯度范数越大越危险 → 边距收缩
      · desire / fear：被 learning 信号推到与 computed desire/fear 一致
    """
    deltas: Dict[str, float] = {}

    def spec_for(name):
        return state.specs.get(name)

    # energy: 恒耗
    if spec_for("energy"):
        deltas["energy"] = -energy_cost

    # resources: 由 loss 下降驱动
    if spec_for("resources") and "loss" in metrics:
        dl = metrics.get("loss_delta", 0.0)          # 本步损失变化（负=下降）
        deltas["resources"] = -0.6 * dl + 0.001      # loss 降 → resources 升

    # consistency: 由损失方差（抖动）反向驱动
    if spec_for("consistency") and "loss_var" in metrics:
        deltas["consistency"] = -metrics["loss_var"] * 0.5 + 0.002

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
    state.step(compute_state_deltas(state, sig,
                                    {"loss": 4.0, "loss_delta": -0.3, "loss_var": 0.02,
                                     "grad_norm": 0.8}))
    print("rewards OK: reward=", round(intrinsic_reward(sig, cfg), 4),
          "loss=", round(float(drives_loss(sig, cfg)), 4),
          "state=", {k: round(v, 3) for k, v in state.snapshot().items()})
