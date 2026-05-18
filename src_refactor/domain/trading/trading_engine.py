from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.types import Fill, MarketExecutionSnapshot, OrderRequest
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


@dataclass(slots=True)
class TradingEngine:
    broker: BrokerGateway
    portfolio: PortfolioManager
    risk: RiskManager
    config: TradingEngineConfig = field(default_factory=TradingEngineConfig)

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
        first_timestamp = min((snapshot.current_timestamp for snapshot in snapshots.values()), default=None)
        equity = self.portfolio.record_equity(first_timestamp, mark_prices) if first_timestamp is not None else None

        for symbol, snapshot in snapshots.items():
            position = self.portfolio.position_snapshot(symbol)
            if position is None:
                continue
            fill = self.broker.resolve_position_exit(position, snapshot)
            if fill is None:
                continue
            fills.append(fill)
            trade = self.portfolio.close_position(
                symbol=symbol,
                exit_price=fill.price,
                reason=fill.reason,
                closed_at=fill.timestamp,
            )
            if trade is not None:
                closed_trades.append(trade)
                self.risk.record_trade_close(
                    symbol=symbol,
                    pnl_pct=trade.pnl_pct,
                    reason=trade.reason,
                    bar_index=bar_index,
                )

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
                open_positions_count=len(self.portfolio.state.positions),
            ):
                continue
            if candidate.symbol in self.portfolio.state.positions:
                continue
            stop_pct, take_pct = self._barrier_pcts(candidate)
            if stop_pct is None or take_pct is None:
                continue
            sizing = self.risk.size_position(
                balance=self.portfolio.balance,
                available_balance=self.portfolio.available_balance,
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

        return TradingEngineStepResult(
            fills=tuple(fills),
            opened_orders=tuple(opened_orders),
            closed_trades=tuple(closed_trades),
            equity=equity,
        )

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
