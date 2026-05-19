from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src_refactor.core.contracts import MarketBatchStream
from src_refactor.core.types import Candle, MarketDataBatch


class CandleRepository(Protocol):
    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ...


@dataclass(frozen=True, slots=True)
class HistoricalFrameMarketStream(MarketBatchStream):
    frame: pd.DataFrame
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    def stream(self) -> Iterator[MarketDataBatch]:
        required = {self.timestamp_column, self.symbol_column, "open", "high", "low", "close", "volume"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Market frame is missing required columns: {sorted(missing)}")

        prepared = self.frame.copy()
        prepared[self.timestamp_column] = pd.to_datetime(prepared[self.timestamp_column])
        prepared = prepared.sort_values([self.timestamp_column, self.symbol_column])
        for timestamp, group in prepared.groupby(self.timestamp_column, sort=True):
            candles = tuple(
                Candle(
                    symbol=str(row[self.symbol_column]),
                    timeframe=str(row.get("timeframe", "")),
                    timestamp=pd.to_datetime(row[self.timestamp_column]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for _, row in group.iterrows()
            )
            yield MarketDataBatch(timestamp=pd.to_datetime(timestamp), candles=candles)


@dataclass(frozen=True, slots=True)
class HistoricalCandleFrameLoader:
    repository: CandleRepository
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    timeframe_column: str = "timeframe"

    def load_symbol(self, symbol: str, timeframe: str) -> pd.DataFrame:
        frame = self.repository.load_candles(symbol, timeframe)
        return normalize_candle_frame(
            frame,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
            timeframe_column=self.timeframe_column,
        )

    def load_symbols(self, symbols: tuple[str, ...] | list[str], timeframe: str) -> pd.DataFrame:
        frames = [self.load_symbol(symbol, timeframe) for symbol in symbols]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return empty_candle_frame(
                timestamp_column=self.timestamp_column,
                symbol_column=self.symbol_column,
                timeframe_column=self.timeframe_column,
            )
        return pd.concat(frames, ignore_index=True).sort_values(
            [self.timestamp_column, self.symbol_column]
        ).reset_index(drop=True)


def normalize_candle_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
    timeframe_column: str = "timeframe",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_candle_frame(
            timestamp_column=timestamp_column,
            symbol_column=symbol_column,
            timeframe_column=timeframe_column,
        )

    required = {timestamp_column, "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Candle frame is missing required columns: {missing}")

    output = frame.copy()
    output[timestamp_column] = _normalize_timestamp(output[timestamp_column])
    output[symbol_column] = symbol
    output[timeframe_column] = timeframe
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["close_time"] = output[timestamp_column] + pd.to_timedelta(_timeframe_to_ms(timeframe), unit="ms")
    output = output.dropna(subset=[timestamp_column, "open", "high", "low", "close", "volume"])
    columns = [
        timestamp_column,
        symbol_column,
        timeframe_column,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    ]
    return output.loc[:, columns].sort_values(timestamp_column).reset_index(drop=True)


def empty_candle_frame(
    *,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
    timeframe_column: str = "timeframe",
) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            timestamp_column,
            symbol_column,
            timeframe_column,
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ]
    )


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms")
    return pd.to_datetime(series)


def _timeframe_to_ms(timeframe: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    unit = timeframe[-1:]
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[unit]
