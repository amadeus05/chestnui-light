from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.types import AccountSnapshot, Fill, MarketExecutionSnapshot, OrderRequest, OrderSnapshot
from src_refactor.domain.execution import ExecutionJournal
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager, Trade
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalCandidate
from src_refactor.domain.trading.order_factory import OrderFactory


@dataclass(frozen=True, slots=True)
class TradingEngineConfig:
    max_new_positions_per_bar: int = 1


@dataclass(frozen=True, slots=True)
class TradingEngineStepResult:
    fills: tuple[Fill, ...] = ()
    opened_orders: tuple[OrderRequest, ...] = ()
    accepted_orders: tuple[OrderSnapshot, ...] = ()
    rejected_orders: tuple[OrderSnapshot, ...] = ()
    closed_trades: tuple[Trade, ...] = ()
    equity: float | None = None
    account: AccountSnapshot | None = None


@dataclass(slots=True)
class TradingEngine:
    broker: BrokerGateway
    portfolio: PortfolioManager
    risk: RiskManager
    config: TradingEngineConfig = field(default_factory=TradingEngineConfig)
    order_factory: OrderFactory = field(default_factory=OrderFactory)
    execution_journal: ExecutionJournal | None = None
    _last_mark_prices: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        set_provider = getattr(self.broker, "set_account_snapshot_provider", None)
        if callable(set_provider):
            set_provider(lambda: self.portfolio.account_snapshot(self._last_mark_prices))

    def on_market_batch(
        self,
        *,
        bar_index: int,
        snapshots: dict[str, MarketExecutionSnapshot],
        candidates: list[SignalCandidate],
    ) -> TradingEngineStepResult:
        fills: list[Fill] = []
        closed_trades: list[Trade] = []

        mark_prices = {symbol: snapshot.current_close for symbol, snapshot in snapshots.items()}
        self._last_mark_prices = dict(mark_prices)
        first_timestamp = min((snapshot.current_timestamp for snapshot in snapshots.values()), default=None)
        first_execution_timestamp = min((snapshot.next_timestamp for snapshot in snapshots.values()), default=None)
        if first_execution_timestamp is not None:
            self.risk.on_bar(first_execution_timestamp)

        account = self.broker.get_account_snapshot()
        if first_timestamp is not None:
            self.portfolio.record_equity(first_timestamp, mark_prices)
        equity = account.equity

        for symbol, snapshot in snapshots.items():
            position = account.positions.get(symbol) or self.portfolio.position_snapshot(symbol)
            if position is None:
                continue
            fill = self.broker.resolve_position_exit(position, snapshot)
            if fill is None:
                continue
            fills.append(fill)
            self._record_fill(fill)
            trade = self._close_position_from_fill(fill, bar_index)
            if trade is not None:
                closed_trades.append(trade)
                self._record_trade_closed(trade)

        account = self.broker.get_account_snapshot()

        opened_orders: list[OrderRequest] = []
        accepted_orders: list[OrderSnapshot] = []
        rejected_orders: list[OrderSnapshot] = []
        opened_this_bar = 0
        for candidate in candidates:
            if opened_this_bar >= self.config.max_new_positions_per_bar:
                break
            snapshot = snapshots.get(candidate.symbol)
            if snapshot is None:
                continue
            if not self.risk.can_open_symbol(
                candidate.symbol,
                bar_index,
                open_positions_count=len(account.positions),
            ):
                continue
            if candidate.symbol in account.positions or candidate.symbol in self.portfolio.state.positions:
                continue
            sizing = self.risk.size_position(
                balance=account.balance,
                available_balance=max(0.0, account.balance - account.used_margin),
                stop_pct=candidate.stop_pct,
            )
            if sizing is None:
                continue

            order = self.order_factory.from_signal_candidate(
                candidate,
                position_notional=sizing.position_notional,
                snapshot=snapshot,
            )
            self._record_order_submitted(order)
            result = self.broker.place_order(order)
            if not result.accepted:
                if result.order is not None:
                    rejected_orders.append(result.order)
                    self._record_order_rejected(result.order)
                continue
            opened_orders.append(order)
            if result.order is not None:
                accepted_orders.append(result.order)
                self._record_order_accepted(result.order)

            order_fills = list(result.fills)
            order_fills.extend(self.broker.process_market_snapshot(snapshot))
            for fill in order_fills:
                fills.append(fill)
                self._record_fill(fill)
                if fill.reason != "ENTRY":
                    continue
                self.portfolio.open_position(
                    symbol=fill.symbol,
                    direction=candidate.direction,
                    entry_price=fill.price,
                    position_notional=sizing.position_notional,
                    required_margin=sizing.required_margin,
                    stop_pct=candidate.stop_pct,
                    take_pct=candidate.take_pct,
                    opened_at=fill.timestamp,
                )
                opened_this_bar += 1
                account = self.broker.get_account_snapshot()

                position = self.portfolio.position_snapshot(fill.symbol)
                if position is None:
                    continue
                exit_fill = self.broker.resolve_position_exit(position, snapshot)
                if exit_fill is None:
                    continue
                fills.append(exit_fill)
                self._record_fill(exit_fill)
                trade = self._close_position_from_fill(exit_fill, bar_index)
                if trade is not None:
                    closed_trades.append(trade)
                    self._record_trade_closed(trade)
                    account = self.broker.get_account_snapshot()

        self._record_account_snapshot(account, first_timestamp)

        return TradingEngineStepResult(
            fills=tuple(fills),
            opened_orders=tuple(opened_orders),
            accepted_orders=tuple(accepted_orders),
            rejected_orders=tuple(rejected_orders),
            closed_trades=tuple(closed_trades),
            equity=equity,
            account=account,
        )

    def _close_position_from_fill(self, fill: Fill, bar_index: int) -> Trade | None:
        trade = self.portfolio.close_position_from_fill(fill)
        if trade is not None:
            self.risk.record_trade(trade, bar_index=bar_index)
        return trade

    def close_all_positions(
        self,
        *,
        timestamp: pd.Timestamp,
        mark_prices: dict[str, float],
        reason: str = "FINAL",
    ) -> TradingEngineStepResult:
        fills: list[Fill] = []
        closed_trades: list[Trade] = []
        self._last_mark_prices = dict(mark_prices)
        account = self.broker.get_account_snapshot()
        positions = dict(account.positions)
        for symbol in self.portfolio.state.positions:
            if symbol not in positions:
                position = self.portfolio.position_snapshot(symbol)
                if position is not None:
                    positions[symbol] = position

        for symbol, position in positions.items():
            mark_price = mark_prices.get(symbol)
            if mark_price is None or not np.isfinite(mark_price):
                continue
            resolve_mark_close = getattr(self.broker, "resolve_position_mark_close", None)
            if not callable(resolve_mark_close):
                continue
            fill = resolve_mark_close(
                position,
                timestamp=pd.to_datetime(timestamp),
                mark_price=float(mark_price),
                reason=reason,
            )
            fills.append(fill)
            self._record_fill(fill)
            trade = self.portfolio.close_position_from_fill(fill)
            if trade is not None:
                closed_trades.append(trade)
                self._record_trade_closed(trade)

        self.portfolio.record_equity(pd.to_datetime(timestamp), mark_prices)
        account = self.broker.get_account_snapshot()
        self._record_account_snapshot(account, pd.to_datetime(timestamp))
        return TradingEngineStepResult(
            fills=tuple(fills),
            closed_trades=tuple(closed_trades),
            equity=account.equity,
            account=account,
        )

    def _record_order_submitted(self, order: OrderRequest) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_order_submitted(order)

    def _record_order_accepted(self, order: OrderSnapshot) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_order_accepted(order)

    def _record_order_rejected(self, order: OrderSnapshot) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_order_rejected(order)

    def _record_fill(self, fill: Fill) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_fill(fill)

    def _record_trade_closed(self, trade: Trade) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_trade_closed(trade)

    def _record_account_snapshot(self, account: AccountSnapshot, timestamp: pd.Timestamp | None) -> None:
        if self.execution_journal is not None:
            self.execution_journal.record_account_snapshot(account, timestamp=timestamp)
