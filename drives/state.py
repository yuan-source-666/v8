"""持续内部状态 S(t) —— 驱动信号层的载体。

设计（对齐架构文档 v1.2 的 L2 稳态驱动机制）：
  内部状态由若干标量分量组成，例如：
    resources     资源水平（如有效可学习信号的多寡）
    energy        能量 / 可用算力预算
    consistency   一致性（模型近期输出的稳定性，越低越“涣散”）
    safety_margin 安全边距（距崩溃/越界的余量）
    desire        欲望水平（趋近目标状态的程度）
    fear          恐惧水平（越出安全边界的前瞻风险）

每个分量带 [low, high] 硬边界（安全边界）与 setpoint（稳态设定值），
以及 step（单次更新允许的最大速度）。更新时先限制速度、再 clamp 到边界，
保证 S(t) 任何时刻都在安全区间内 —— 这是“稳态”的物理前提。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StateSpec:
    """单个状态分量的规格。"""
    setpoint: float = 0.5
    low: float = 0.0
    high: float = 1.0
    step: float = 0.02          # 单步更新最大速度（L2 稳态限制）

    def clamp(self, v: float) -> float:
        return min(max(v, self.low), self.high)


def parse_specs(components: Dict[str, dict]) -> Dict[str, StateSpec]:
    """把 YAML 里的分量定义（setpoint/low/high，可含 step）解析成 StateSpec。"""
    specs = {}
    for name, d in components.items():
        specs[name] = StateSpec(
            setpoint=float(d.get("setpoint", 0.5)),
            low=float(d.get("low", 0.0)),
            high=float(d.get("high", 1.0)),
            step=float(d.get("step", 0.02)),
        )
    return specs


class InternalState:
    """S(t)：带边界、带速度限制、可由信号驱动的内部状态表。"""

    def __init__(self, specs: Dict[str, StateSpec],
                 init_values: Optional[Dict[str, float]] = None):
        self.specs = specs
        self.values: Dict[str, float] = {}
        for name, spec in specs.items():
            v = spec.setpoint
            if init_values and name in init_values:
                v = init_values[name]
            self.values[name] = spec.clamp(v)
        self._step_counter = 0

    # ------------------------------------------------------------------
    def get(self, name: str) -> float:
        return self.values[name]

    def snapshot(self) -> Dict[str, float]:
        """返回当前状态的拷贝（便于记录日志）。"""
        return dict(self.values)

    # ------------------------------------------------------------------
    def step(self, deltas: Dict[str, float]) -> Dict[str, float]:
        """按信号 deltas 推进状态：先限速（不超过 spec.step），再 clamp 边界。

        Returns:
            更新后的状态快照。
        """
        for name, d in deltas.items():
            if name not in self.specs:
                continue
            spec = self.specs[name]
            clamped_d = max(-spec.step, min(spec.step, float(d)))
            self.values[name] = spec.clamp(self.values[name] + clamped_d)
        self._step_counter += 1
        return self.values

    # ------------------------------------------------------------------
    def predict_next(self, deltas: Dict[str, float], steps: int = 1) -> Dict[str, float]:
        """前瞻：预测把 deltas 连续外推 steps 步后的状态（用于恐惧的前向惩罚）。

        返回与 values 同构的“预测下一状态”（不受限速/边界约束，保留越界量，
        这正是恐惧信号要惩罚的“危险趋势”）。
        """
        pred = {}
        for name, v in self.values.items():
            d = deltas.get(name, 0.0)
            pred[name] = v + d * steps
        return pred


if __name__ == "__main__":
    specs = parse_specs({
        "energy": {"setpoint": 0.8, "low": 0.0, "high": 1.0, "step": 0.05},
        "fear": {"setpoint": 0.3, "low": 0.0, "high": 1.0, "step": 0.05},
    })
    s = InternalState(specs)
    s.step({"energy": -2.0})      # 一次掉 2.0 → 被限速到 -0.05
    assert s.get("energy") == 0.75
    s.step({"fear": 10.0})
    assert s.get("fear") == 0.35
    print("InternalState OK:", s.snapshot(), "predict_next=", s.predict_next({"energy": -0.05}, 3))
