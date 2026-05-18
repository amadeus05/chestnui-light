from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market"]
OrderStatus = Literal["open", "filled", "cancelled", "rejected"]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    reduce_only: bool = False
    stop_pct: float | None = None
    take_pct: float | None = None
    created_at: pd.Timestamp | None = None
    status: OrderStatus = "open"

    @property
    def direction(self) -> int:
        return 1 if self.side == "buy" else -1


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    reduce_only: bool = False
    stop_pct: float | None = None
    take_pct: float | None = None
    created_at: pd.Timestamp | None = None
    status: OrderStatus = "open"
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None

    @classmethod
    def from_request(
        cls,
        order: OrderRequest,
        *,
        status: OrderStatus | None = None,
        filled_quantity: float = 0.0,
        avg_fill_price: float | None = None,
    ) -> "OrderSnapshot":
        return cls(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            reduce_only=order.reduce_only,
            stop_pct=order.stop_pct,
            take_pct=order.take_pct,
            created_at=order.created_at,
            status=status or order.status,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
        )

    @property
    def direction(self) -> int:
        return 1 if self.side == "buy" else -1
