from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src_refactor.domain.portfolio.portfolio_manager import Trade


@dataclass(frozen=True, slots=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    leverage: float = 1.0
    min_position_notional: float = 10.0
    max_open_positions: int = 1
    sl_cooldown_bars: int = 0
    max_sl_per_day: int = 0
    reduce_risk_after_consecutive_losses: int = 0
    reduced_risk_per_trade: float | None = None


@dataclass(frozen=True, slots=True)
class PositionSizing:
    position_notional: float
    required_margin: float
    risk_per_trade: float


@dataclass(slots=True)
class RiskState:
    consecutive_loss_count: int = 0
    daily_sl_count: int = 0
    current_trade_day: pd.Timestamp | None = None
    stop_cooldown_until_index: dict[str, int] = field(default_factory=dict)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None, state: RiskState | None = None) -> None:
        self.config = config or RiskConfig()
        self.state = state or RiskState()

    def effective_risk_per_trade(self) -> float:
        threshold = self.config.reduce_risk_after_consecutive_losses
        reduced = self.config.reduced_risk_per_trade
        if (
            threshold > 0
            and reduced is not None
            and 0 < reduced < self.config.risk_per_trade
            and self.state.consecutive_loss_count >= threshold
        ):
            return reduced
        return self.config.risk_per_trade

    def on_bar(self, timestamp: pd.Timestamp) -> None:
        trade_day = pd.to_datetime(timestamp).normalize()
        if self.state.current_trade_day is None or trade_day != self.state.current_trade_day:
            self.state.current_trade_day = trade_day
            self.reset_daily_limits()

    def can_open_symbol(self, symbol: str, bar_index: int, open_positions_count: int) -> bool:
        if open_positions_count >= self.config.max_open_positions:
            return False
        if self.config.max_sl_per_day > 0 and self.state.daily_sl_count >= self.config.max_sl_per_day:
            return False
        return bar_index >= self.state.stop_cooldown_until_index.get(symbol, -1)

    def size_position(
        self,
        *,
        balance: float,
        available_balance: float,
        stop_pct: float,
    ) -> PositionSizing | None:
        if stop_pct <= 0 or available_balance <= 0:
            return None
        effective_risk = self.effective_risk_per_trade()
        risk_capital = balance * effective_risk
        position_notional = min(risk_capital / stop_pct, balance * self.config.leverage)
        required_margin = min(position_notional / self.config.leverage, available_balance)
        position_notional = min(position_notional, required_margin * self.config.leverage)
        if position_notional < self.config.min_position_notional or required_margin <= 0:
            return None
        return PositionSizing(
            position_notional=position_notional,
            required_margin=required_margin,
            risk_per_trade=effective_risk,
        )

    def record_trade_close(self, *, symbol: str, pnl_pct: float, reason: str, bar_index: int) -> None:
        if pnl_pct > 0:
            self.state.consecutive_loss_count = 0
        else:
            self.state.consecutive_loss_count += 1

        if reason == "SL":
            self.state.daily_sl_count += 1
            if self.config.sl_cooldown_bars > 0:
                self.state.stop_cooldown_until_index[symbol] = bar_index + self.config.sl_cooldown_bars

    def record_trade(self, trade: Trade, *, bar_index: int) -> None:
        self.record_trade_close(
            symbol=trade.symbol,
            pnl_pct=trade.pnl_pct,
            reason=trade.reason,
            bar_index=bar_index,
        )

    def reset_daily_limits(self) -> None:
        self.state.daily_sl_count = 0
