"""
Сайзинг входа как в bt.py (строки 947–954, 1013–1015) и paper.py (395–397).
"""
from __future__ import annotations

from .models.risk_limits import RiskLimits


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def effective_risk_per_trade(self, consecutive_loss_count: int) -> float:
        effective = self._limits.risk_per_trade
        if (
            self._limits.reduce_risk_after_consecutive_losses > 0
            and self._limits.reduced_risk_per_trade > 0
            and self._limits.reduced_risk_per_trade < self._limits.risk_per_trade
            and consecutive_loss_count >= self._limits.reduce_risk_after_consecutive_losses
        ):
            effective = self._limits.reduced_risk_per_trade
        return effective

    def raw_entry_notional_and_margin(
        self,
        snapshot_balance: float,
        stop_pct: float,
        consecutive_loss_count: int,
    ) -> tuple[float, float]:
        er = self.effective_risk_per_trade(consecutive_loss_count)
        risk_capital = float(snapshot_balance) * er
        position_notional = min(
            risk_capital / float(stop_pct),
            float(snapshot_balance) * self._limits.leverage,
        )
        required_margin = position_notional / self._limits.leverage
        return position_notional, required_margin

    def passes_min_notional(self, position_notional: float) -> bool:
        return float(position_notional) >= self._limits.min_position_notional
