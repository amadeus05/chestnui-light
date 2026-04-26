"""Шаг портфельного бэктеста: снимок на закрытии decision_ts, исполнение на свече exec_ts."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.contracts.data_feed import Bar


@dataclass(frozen=True)
class PortfolioReplayStep:
    """Один шаг: mark по закрытию decision_ts, TP/SL/вход по OHLC exec_ts (следующий бар)."""

    decision_ts: pd.Timestamp
    exec_ts: pd.Timestamp
    mark_prices: dict[str, float]
    exec_bars: dict[str, Bar]
