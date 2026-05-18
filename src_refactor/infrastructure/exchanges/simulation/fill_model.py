from __future__ import annotations

from dataclasses import dataclass

from src_refactor.core.types import Fill, MarketExecutionSnapshot, OrderRequest, PositionSnapshot
from src_refactor.domain.execution import ExecutionPricingConfig, apply_entry_slippage, resolve_trade_exit


@dataclass(frozen=True, slots=True)
class CandleFillModel:
    pricing: ExecutionPricingConfig

    def entry_price(self, direction: int, snapshot: MarketExecutionSnapshot) -> float:
        return apply_entry_slippage(direction, snapshot.next_open, self.pricing)

    def fill_market_order(self, order: OrderRequest, snapshot: MarketExecutionSnapshot) -> Fill:
        price = self.entry_price(order.direction, snapshot)
        fee = abs(order.quantity * price) * self.pricing.taker_fee
        return Fill(
            fill_id=f"{order.order_id}:entry",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=order.quantity,
            fee=fee,
            reason="ENTRY",
            timestamp=snapshot.next_timestamp,
        )

    def resolve_position_exit(
        self,
        *,
        order_id: str,
        position: PositionSnapshot,
        snapshot: MarketExecutionSnapshot,
    ) -> Fill | None:
        if position.stop_pct is None or position.take_pct is None:
            return None
        exit_result = resolve_trade_exit(
            position.direction,
            position.entry_price,
            snapshot.next_open,
            snapshot.next_high,
            snapshot.next_low,
            position.stop_pct,
            position.take_pct,
            self.pricing,
        )
        if exit_result.price is None or exit_result.reason is None:
            return None
        side = "sell" if position.direction == 1 else "buy"
        fee = abs(position.quantity * exit_result.price) * self.pricing.taker_fee
        return Fill(
            fill_id=f"{order_id}:exit:{snapshot.next_timestamp.value}",
            order_id=order_id,
            symbol=position.symbol,
            side=side,
            price=exit_result.price,
            quantity=position.quantity,
            fee=fee,
            reason=exit_result.reason,
            timestamp=snapshot.next_timestamp,
        )
