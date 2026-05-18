from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.core.types.orders import OrderSide


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    fee: float
    reason: str
    timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class MarketExecutionSnapshot:
    symbol: str
    current_timestamp: pd.Timestamp
    next_timestamp: pd.Timestamp
    current_close: float
    next_open: float
    next_high: float
    next_low: float
