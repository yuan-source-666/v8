"""稳态偏差信号 D = |S(t) − setpoint|，并据此计算欲望 / 恐惧驱动信号。

对齐架构文档 v1.2 的 L2 稳态驱动机制：
  · 偏差（deviation）：D_i = |S_i(t) − setpoint_i|，衡量当前状态离稳态有多远。
  · 欲望（desire）：趋近“目标状态”的梯度信号。目标状态 target 由配置给定，
    desire 正比于当前状态到目标状态的归一化距离 —— 距离越大，“想要”越强；
    同时给出带符号的方向 (target − current)，供 rewards 决定往哪个方向推状态。
  · 恐惧（fear）：越出安全边界的前瞻惩罚信号。把当前状态按【近期实际速度】
    （state.velocity()，ΔS 的 EMA）外推 anticipation_steps 步（predict_next），
    若预测值越过 [low, high] 边界、或进入边界内侧 fear_margin 预警带，
    则产生惩罚；越得越多、越接近边界，恐惧越大。
    （注意：趋势必须是真实速度——“朝 setpoint 走一步”的伪趋势在 setpoint
    位于边界内时永远推不出边界，恐惧将恒为 0。）

恐惧与欲望形成“趋利避害”的合力：欲望把系统推向目标，恐惧把系统拦在边界内。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .state import InternalState, StateSpec


class DriveSignals:
    def __init__(self, state: InternalState,
                 drives_cfg: Dict,
                 safety_bound: Optional[Dict] = None,
                 target_state: Optional[Dict[str, float]] = None):
        self.state = state
        self.drives_cfg = drives_cfg or {}
        sb = safety_bound or drives_cfg.get("safety_bound", {})
        self.fear_threshold = float(sb.get("fear_threshold", 0.8))
        self.anticipation_steps = int(sb.get("anticipation_steps", 3))
        self.fear_margin = float(sb.get("fear_margin", 0.10))  # 边界预警带宽度（区间占比）
        # 目标状态（欲望的指向）；缺省默认取各分量 setpoint
        self.target_state: Dict[str, float] = dict(
            target_state or drives_cfg.get("target_state", {}))

    # ------------------------------------------------------------------
    def _target_for(self, name: str) -> float:
        if name in self.target_state:
            return float(self.target_state[name])
        return self.state.specs[name].setpoint

    # ------------------------------------------------------------------
    def deviation(self) -> Dict[str, float]:
        """D = |S(t) − setpoint|，逐分量稳态偏差。"""
        return {n: abs(v - self.state.specs[n].setpoint)
                for n, v in self.state.values.items()}

    def total_deviation(self, power: float = 2.0) -> float:
        """所有分量偏差的 2-范数（归一化），用于 rewards 的正则项。"""
        D = self.deviation()
        n = max(1, len(D))
        return (sum(d ** power for d in D.values()) / n) ** (1.0 / power)

    # ------------------------------------------------------------------
    def desire(self) -> Dict[str, float]:
        """欲望：到目标状态的归一化距离（0..1）。返回 {分量名: 距离}。"""
        out = {}
        for name, spec in self.state.specs.items():
            if name in ("desire", "fear"):
                continue                      # 不做自我递归
            target = self._target_for(name)
            rg = max(spec.high - spec.low, 1e-6)
            out[name] = min(1.0, abs(self.state.get(name) - target) / rg)
        return out

    def desire_total(self) -> float:
        ds = self.desire()
        return sum(ds.values()) / max(1, len(ds)) if ds else 0.0

    def desire_direction(self) -> Dict[str, float]:
        """带符号方向：sign(target − current)，供 rewards 决定状态推动方向。"""
        out = {}
        for name in self.state.specs:
            if name in ("desire", "fear"):
                continue
            dv = self._target_for(name) - self.state.get(name)
            out[name] = 1.0 if dv > 0 else (-1.0 if dv < 0 else 0.0)
        return out

    # ------------------------------------------------------------------
    def fear(self) -> float:
        """恐惧：按近期实际速度外推 anticipation_steps 步，进入边界预警带/越界则惩罚（>=0）。"""
        vel = self.state.velocity()
        penalty = 0.0
        for n, spec in self.state.specs.items():
            if n == "fear":
                continue                      # 恐惧分量不做自我递归
            rg = max(spec.high - spec.low, 1e-6)
            pv = self.state.get(n) + vel.get(n, 0.0) * self.anticipation_steps
            margin = self.fear_margin * rg    # 预警带：距边界 margin 以内即预警
            # 越界惩罚（保持原语义）
            if pv > spec.high:
                penalty += (pv - spec.high) / rg
            # 预警带惩罚：趋势逼近上边界
            elif pv > spec.high - margin:
                penalty += (pv - (spec.high - margin)) / rg
            if pv < spec.low:
                penalty += (spec.low - pv) / rg
            elif pv < spec.low + margin:
                penalty += ((spec.low + margin) - pv) / rg
        return min(1.0, penalty)

    def safety_critical(self) -> bool:
        """是否已触发安全边界预警（供 train.py 记录日志 / 触发保护）。"""
        return self.fear() >= self.fear_threshold

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """汇总输出本轮全部 L2 信号，供 rewards 与日志使用。"""
        return {
            "deviation": self.total_deviation(),
            "desire": self.desire_total(),
            "fear": self.fear(),
            "critical": 1.0 if self.safety_critical() else 0.0,
        }


if __name__ == "__main__":
    from .state import InternalState, parse_specs
    specs = parse_specs({
        "resources": {"setpoint": 0.75, "low": 0.0, "high": 1.0, "step": 0.02},
        "energy": {"setpoint": 0.8, "low": 0.0, "high": 1.0, "step": 0.02},
        "safety_margin": {"setpoint": 0.6, "low": 0.0, "high": 1.0, "step": 0.02},
    })
    state = InternalState(specs)
    sig = DriveSignals(state, {}, safety_bound={"anticipation_steps": 5})
    sig.target_state = {"energy": 0.2}
    print("初始信号:", sig.compute())
    # 模拟能量持续下滑逼近下边界 → 恐惧应从 0 升起
    for _ in range(45):
        state.step({"energy": -0.02})
    c = sig.compute()
    assert c["fear"] > 0.0, "恐惧信号应随能量逼近边界而激活"
    print("能量下滑后:", c, "state=", {k: round(v, 3) for k, v in state.snapshot().items()})
    print("signals OK: fear 随真实趋势激活 ✓")
