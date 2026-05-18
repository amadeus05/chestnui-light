from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src_refactor.core.types import AccountSnapshot, PositionSnapshot
from src_refactor.domain.execution import ExecutionPricingConfig, compute_fee_quote, compute_net_pnl_pct


@dataclass(slots=True)
class Position:
    trade_number: int
    symbol: str
    direction: int
    entry_price: float
    size: float
    margin: float
    stop_pct: float
    take_pct: float
    opened_at: pd.Timestamp


@dataclass(frozen=True, slots=True)
class Trade:
    trade_number: int
    symbol: str
    direction: int
    reason: str
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_abs: float
    commission: float


@dataclass(slots=True)
class PortfolioState:
    balance: float
    used_margin: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    next_trade_number: int = 1
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0


class PortfolioManager:
    def __init__(self, initial_balance: float = 100.0, pricing: ExecutionPricingConfig | None = None) -> None:
        self.pricing = pricing or ExecutionPricingConfig()
        self.initial_balance = float(initial_balance)
        self.state = PortfolioState(balance=float(initial_balance), peak_equity=float(initial_balance))

    @property
    def balance(self) -> float:
        return self.state.balance

    @property
    def available_balance(self) -> float:
        return self.state.balance - self.state.used_margin

    def open_position(
        self,
        *,
        symbol: str,
        direction: int,
        entry_price: float,
        position_notional: float,
        required_margin: float,
        stop_pct: float,
        take_pct: float,
        opened_at: pd.Timestamp,
    ) -> Position:
        trade_number = self.state.next_trade_number
        self.state.next_trade_number += 1
        self.state.used_margin += required_margin
        position = Position(
            trade_number=trade_number,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=position_notional,
            margin=required_margin,
            stop_pct=stop_pct,
            take_pct=take_pct,
            opened_at=opened_at,
        )
        self.state.positions[symbol] = position
        return position

    def close_position(
        self,
        *,
        symbol: str,
        exit_price: float,
        reason: str,
        closed_at: pd.Timestamp,
    ) -> Trade | None:
        position = self.state.positions.get(symbol)
        if position is None:
            return None

        pnl_pct = compute_net_pnl_pct(position.direction, position.entry_price, exit_price, self.pricing)
        pnl_abs = position.size * pnl_pct
        commission = compute_fee_quote(position.size, self.pricing) * 2.0
        self.state.used_margin = max(0.0, self.state.used_margin - position.margin)
        self.state.balance += pnl_abs

        trade = Trade(
            trade_number=position.trade_number,
            symbol=symbol,
            direction=position.direction,
            reason=reason,
            opened_at=position.opened_at,
            closed_at=closed_at,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_abs=pnl_abs,
            commission=commission,
        )
        self.state.trades.append(trade)
        del self.state.positions[symbol]
        return trade

    def compute_equity(self, mark_prices: dict[str, float]) -> float:
        equity = float(self.state.balance)
        for symbol, position in self.state.positions.items():
            mark_price = mark_prices.get(symbol)
            if mark_price is None or not np.isfinite(mark_price):
                continue
            pnl_pct = compute_net_pnl_pct(position.direction, position.entry_price, float(mark_price), self.pricing)
            equity += float(position.size) * pnl_pct
        return equity

    def record_equity(self, timestamp: pd.Timestamp, mark_prices: dict[str, float]) -> float:
        equity = self.compute_equity(mark_prices)
        self.state.equity_curve.append((timestamp, equity))
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        if self.state.peak_equity > 0:
            current_dd = (self.state.peak_equity - equity) / self.state.peak_equity * 100
            self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, current_dd)
        return equity

    def account_snapshot(self, mark_prices: dict[str, float] | None = None) -> AccountSnapshot:
        return AccountSnapshot(
            balance=float(self.state.balance),
            equity=self.compute_equity(mark_prices or {}),
            used_margin=float(self.state.used_margin),
            positions={
                symbol: snapshot
                for symbol in self.state.positions
                if (snapshot := self.position_snapshot(symbol)) is not None
            },
        )

    def position_snapshot(self, symbol: str) -> PositionSnapshot | None:
        position = self.state.positions.get(symbol)
        if position is None:
            return None
        return PositionSnapshot(
            symbol=position.symbol,
            direction=position.direction,
            quantity=position.size / position.entry_price,
            entry_price=position.entry_price,
            stop_pct=position.stop_pct,
            take_pct=position.take_pct,
            opened_at=position.opened_at,
        )
