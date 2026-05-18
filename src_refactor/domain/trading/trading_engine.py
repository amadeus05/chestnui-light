from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.types import AccountSnapshot, Fill, MarketExecutionSnapshot, OrderRequest
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager, Trade
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalCandidate


@dataclass(frozen=True, slots=True)
class TradingEngineConfig:
    max_new_positions_per_bar: int = 1


@dataclass(frozen=True, slots=True)
class TradingEngineStepResult:
    fills: tuple[Fill, ...] = ()
    opened_orders: tuple[OrderRequest, ...] = ()
    closed_trades: tuple[Trade, ...] = ()
    equity: float | None = None
    account: AccountSnapshot | None = None


@dataclass(slots=True)
class TradingEngine:
    broker: BrokerGateway
    portfolio: PortfolioManager
    risk: RiskManager
    config: TradingEngineConfig = field(default_factory=TradingEngineConfig)
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
        account = self.broker.get_account_snapshot()
        first_timestamp = min((snapshot.current_timestamp for snapshot in snapshots.values()), default=None)
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
            trade = self._close_position_from_fill(fill, bar_index)
            if trade is not None:
                closed_trades.append(trade)

        account = self.broker.get_account_snapshot()

        opened_orders: list[OrderRequest] = []
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
            stop_pct, take_pct = self._barrier_pcts(candidate)
            if stop_pct is None or take_pct is None:
                continue
            sizing = self.risk.size_position(
                balance=account.balance,
                available_balance=max(0.0, account.balance - account.used_margin),
                stop_pct=stop_pct,
            )
            if sizing is None:
                continue

            order = self._order_from_candidate(candidate, sizing.position_notional, stop_pct, take_pct, snapshot)
            result = self.broker.place_order(order)
            if not result.accepted:
                continue
            opened_orders.append(order)

            order_fills = list(result.fills)
            order_fills.extend(self.broker.process_market_snapshot(snapshot))
            for fill in order_fills:
                fills.append(fill)
                if fill.reason != "ENTRY":
                    continue
                self.portfolio.open_position(
                    symbol=fill.symbol,
                    direction=candidate.direction,
                    entry_price=fill.price,
                    position_notional=sizing.position_notional,
                    required_margin=sizing.required_margin,
                    stop_pct=stop_pct,
                    take_pct=take_pct,
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
                trade = self._close_position_from_fill(exit_fill, bar_index)
                if trade is not None:
                    closed_trades.append(trade)

        return TradingEngineStepResult(
            fills=tuple(fills),
            opened_orders=tuple(opened_orders),
            closed_trades=tuple(closed_trades),
            equity=equity,
            account=account,
        )

    def _close_position_from_fill(self, fill: Fill, bar_index: int) -> Trade | None:
        trade = self.portfolio.close_position(
            symbol=fill.symbol,
            exit_price=fill.price,
            reason=fill.reason,
            closed_at=fill.timestamp,
        )
        if trade is not None:
            self.risk.record_trade_close(
                symbol=fill.symbol,
                pnl_pct=trade.pnl_pct,
                reason=trade.reason,
                bar_index=bar_index,
            )
        return trade

    @staticmethod
    def _barrier_pcts(candidate: SignalCandidate) -> tuple[float | None, float | None]:
        stop_pct = candidate.prediction.raw.get("barrier_stop_pct")
        take_pct = candidate.prediction.raw.get("barrier_take_pct")
        if stop_pct is None or take_pct is None:
            return None, None
        return float(stop_pct), float(take_pct)

    @staticmethod
    def _order_from_candidate(
        candidate: SignalCandidate,
        position_notional: float,
        stop_pct: float,
        take_pct: float,
        snapshot: MarketExecutionSnapshot,
    ) -> OrderRequest:
        side = "buy" if candidate.direction == 1 else "sell"
        quantity = position_notional / snapshot.next_open
        return OrderRequest(
            order_id=f"{candidate.symbol}:{snapshot.next_timestamp.value}:{side}",
            symbol=candidate.symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            stop_pct=stop_pct,
            take_pct=take_pct,
            created_at=snapshot.current_timestamp,
        )
