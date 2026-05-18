from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from src_refactor.core.contracts import MarketBatchStream
from src_refactor.core.types import Candle, MarketDataBatch


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
