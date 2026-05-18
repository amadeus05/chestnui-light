from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

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
class MarketDataBatch:
    timestamp: pd.Timestamp
    candles: tuple[Candle, ...]

    @classmethod
    def from_candles(cls, candles: Iterable[Candle]) -> "MarketDataBatch":
        batch_candles = tuple(candles)
        if not batch_candles:
            raise ValueError("MarketDataBatch requires at least one candle.")
        timestamps = {pd.to_datetime(candle.timestamp) for candle in batch_candles}
        if len(timestamps) != 1:
            raise ValueError("MarketDataBatch candles must share one timestamp.")
        return cls(timestamp=timestamps.pop(), candles=batch_candles)

    @property
    def by_symbol(self) -> dict[str, Candle]:
        return {candle.symbol: candle for candle in self.candles}

    @property
    def mark_prices(self) -> dict[str, float]:
        return {candle.symbol: float(candle.close) for candle in self.candles}


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
