from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

Symbol = str


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketDataEvent:
    candle: Candle


@dataclass(frozen=True, slots=True)
class MarketDataSubscription:
    symbol: str
    timeframe: str


@dataclass(frozen=True, slots=True)
class MarketDataset:
    frame: pd.DataFrame
    timeframe: str
    symbols: tuple[str, ...]
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> "MarketDataset":
        mask = self.frame[self.timestamp_column].between(start, end, inclusive="both")
        return MarketDataset(
            frame=self.frame.loc[mask].copy(),
            timeframe=self.timeframe,
            symbols=self.symbols,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        )
