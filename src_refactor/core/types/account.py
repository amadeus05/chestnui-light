from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.types.orders import OrderSnapshot


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    direction: int
    quantity: float
    entry_price: float
    stop_pct: float | None = None
    take_pct: float | None = None
    opened_at: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    balance: float
    equity: float
    used_margin: float
    positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    open_orders: dict[str, OrderSnapshot] = field(default_factory=dict)
