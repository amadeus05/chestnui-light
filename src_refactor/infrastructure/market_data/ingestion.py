from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts import ExchangeMarketDataClient
from src_refactor.infrastructure.market_data.parquet_store import ParquetMarketDataStore


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    written_rows: dict[str, int] = field(default_factory=dict)

    @property
    def total_written_rows(self) -> int:
        return sum(self.written_rows.values())


@dataclass(frozen=True, slots=True)
class MarketDataIngestionService:
    client: ExchangeMarketDataClient
    store: ParquetMarketDataStore

    def backfill_raw(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        timeframes: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        include_premium_index: bool = True,
        include_funding: bool = True,
        include_open_interest: bool = True,
    ) -> IngestionSummary:
        written: dict[str, int] = {}
        start_ts = _normalize_timestamp(start)
        end_ts = _normalize_timestamp(end)

        for symbol in symbols:
            if include_funding:
                written[f"funding:{symbol}"] = self._sync_funding(symbol, start_ts, end_ts)

            for timeframe in timeframes:
                written[f"candles:{timeframe}:{symbol}"] = self._sync_candles(symbol, timeframe, start_ts, end_ts)
                if include_premium_index:
                    written[f"premium_index:{timeframe}:{symbol}"] = self._sync_premium_index(
                        symbol,
                        timeframe,
                        start_ts,
                        end_ts,
                    )
                if include_open_interest:
                    written[f"open_interest:{timeframe}:{symbol}"] = self._sync_open_interest(
                        symbol,
                        timeframe,
                        start_ts,
                        end_ts,
                    )

        return IngestionSummary(written_rows=written)

    def _sync_candles(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        fetch_start = self._next_start(
            self.store.load_candles(symbol, timeframe),
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        if fetch_start > end:
            return 0
        frame = self.client.fetch_candles(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        return self.store.save_candles(symbol, timeframe, frame)

    def _sync_premium_index(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        fetch_start = self._next_start(
            self.store.load_premium_index_klines(symbol, timeframe),
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        if fetch_start > end:
            return 0
        frame = self.client.fetch_premium_index_klines(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        return self.store.save_premium_index_klines(symbol, timeframe, frame)

    def _sync_funding(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        fetch_start = self._next_start(
            self.store.load_funding_rates(symbol),
            requested_start=start,
            step_ms=1,
        )
        if fetch_start > end:
            return 0
        frame = self.client.fetch_funding_rates(symbol=symbol, start=fetch_start, end=end)
        return self.store.save_funding_rates(symbol, frame)

    def _sync_open_interest(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        fetch_start = self._next_start(
            self.store.load_open_interest(symbol, timeframe),
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        if fetch_start > end:
            return 0
        frame = self.client.fetch_open_interest(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        return self.store.save_open_interest(symbol, timeframe, frame)

    @staticmethod
    def _next_start(frame: pd.DataFrame, *, requested_start: pd.Timestamp, step_ms: int) -> pd.Timestamp:
        if frame.empty:
            return requested_start
        latest = pd.to_datetime(frame["timestamp"], errors="coerce").max()
        if pd.isna(latest):
            return requested_start
        latest = _normalize_timestamp(latest)
        return max(requested_start, latest + pd.to_timedelta(step_ms, unit="ms"))


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.to_datetime(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)
