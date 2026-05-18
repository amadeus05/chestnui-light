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
