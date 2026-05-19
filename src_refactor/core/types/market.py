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
        mask = self.frame[self.timestamp_column].between(pd.to_datetime(start), pd.to_datetime(end), inclusive="both")
        return MarketDataset(
            frame=self.frame.loc[mask].copy(),
            timeframe=self.timeframe,
            symbols=self.symbols,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        )

    def with_symbols(self, symbols: tuple[str, ...] | list[str]) -> "MarketDataset":
        symbol_set = set(symbols)
        frame = self.frame.loc[self.frame[self.symbol_column].isin(symbol_set)].copy()
        kept_symbols = tuple(symbol for symbol in self.symbols if symbol in symbol_set)
        return MarketDataset(
            frame=frame,
            timeframe=self.timeframe,
            symbols=kept_symbols,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        )

    def symbols_with_min_rows(
        self,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        min_rows: int = 2,
    ) -> tuple[str, ...]:
        frame = self.frame
        if start is not None:
            frame = frame.loc[frame[self.timestamp_column] >= pd.to_datetime(start)]
        if end is not None:
            frame = frame.loc[frame[self.timestamp_column] <= pd.to_datetime(end)]
        counts = frame.groupby(self.symbol_column).size()
        return tuple(symbol for symbol in self.symbols if int(counts.get(symbol, 0)) >= min_rows)

    def drop_symbols_with_insufficient_rows(
        self,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        min_rows: int = 2,
    ) -> "MarketDataset":
        return self.with_symbols(
            self.symbols_with_min_rows(start=start, end=end, min_rows=min_rows)
        )
