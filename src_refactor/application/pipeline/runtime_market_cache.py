from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.types import Candle, MarketExecutionSnapshot


@dataclass(frozen=True, slots=True)
class MarketContext:
    bar_index: int
    symbol: str
    history: tuple[Candle, ...]
    snapshot: MarketExecutionSnapshot


@dataclass(slots=True)
class RuntimeMarketCache:
    candles: dict[str, list[Candle]] = field(default_factory=dict)

    def update(self, candle: Candle) -> MarketContext | None:
        history = self.candles.setdefault(candle.symbol, [])
        history.append(candle)
        if len(history) < 2:
            return None
        previous = history[-2]
        current = history[-1]
        return MarketContext(
            bar_index=len(history) - 2,
            symbol=candle.symbol,
            history=tuple(history[:-1]),
            snapshot=build_execution_snapshot(previous, current),
        )

    def update_from_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "timestamp",
        symbol_column: str = "symbol",
    ) -> list[MarketContext]:
        required = {timestamp_column, symbol_column, "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Market frame is missing required columns: {sorted(missing)}")

        contexts: list[MarketContext] = []
        for _, row in frame.sort_values([timestamp_column, symbol_column]).iterrows():
            context = self.update(
                Candle(
                    symbol=str(row[symbol_column]),
                    timeframe=str(row.get("timeframe", "")),
                    timestamp=pd.to_datetime(row[timestamp_column]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
            if context is not None:
                contexts.append(context)
        return contexts


def build_execution_snapshot(previous: Candle, current: Candle) -> MarketExecutionSnapshot:
    if previous.symbol != current.symbol:
        raise ValueError("Cannot build execution snapshot from different symbols.")
    return MarketExecutionSnapshot(
        symbol=previous.symbol,
        current_timestamp=pd.to_datetime(previous.timestamp),
        next_timestamp=pd.to_datetime(current.timestamp),
        current_close=float(previous.close),
        next_open=float(current.open),
        next_high=float(current.high),
        next_low=float(current.low),
    )


def candles_to_frame(candles: tuple[Candle, ...] | list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": candle.timestamp,
                "symbol": candle.symbol,
                "timeframe": candle.timeframe,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )
