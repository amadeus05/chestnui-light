from __future__ import annotations

from dataclasses import dataclass, field
import logging

import pandas as pd

from src_refactor.core.contracts import ExchangeMarketDataClient
from src_refactor.infrastructure.market_data.parquet_store import ParquetMarketDataStore


logger = logging.getLogger(__name__)


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
        existing = self.store.load_candles(symbol, timeframe)
        fetch_start = self._next_start(
            existing,
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        self._log_sync_plan(symbol, timeframe, existing, fetch_start, end)
        if fetch_start > end:
            logger.info("[%s-%s] data is already loaded up to %s", symbol, timeframe, _format_ts(end))
            return 0
        frame = self.client.fetch_candles(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        written = self.store.save_candles(symbol, timeframe, frame)
        logger.info("%s %s: %s new candles", symbol, timeframe, written)
        return written

    def _sync_premium_index(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        existing = self.store.load_premium_index_klines(symbol, timeframe)
        fetch_start = self._next_start(
            existing,
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        self._log_sync_plan(symbol, f"premium-{timeframe}", existing, fetch_start, end)
        if fetch_start > end:
            logger.info("[%s-premium-%s] data is already loaded up to %s", symbol, timeframe, _format_ts(end))
            return 0
        frame = self.client.fetch_premium_index_klines(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        written = self.store.save_premium_index_klines(symbol, timeframe, frame)
        logger.info("%s premium index %s: %s new candles", symbol, timeframe, written)
        return written

    def _sync_funding(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        existing = self.store.load_funding_rates(symbol)
        fetch_start = self._next_start(
            existing,
            requested_start=start,
            step_ms=1,
        )
        self._log_sync_plan(symbol, "funding", existing, fetch_start, end)
        if fetch_start > end:
            logger.info("[%s-funding] data is already loaded up to %s", symbol, _format_ts(end))
            return 0
        frame = self.client.fetch_funding_rates(symbol=symbol, start=fetch_start, end=end)
        written = self.store.save_funding_rates(symbol, frame)
        logger.info("%s funding: %s new points", symbol, written)
        return written

    def _sync_open_interest(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        existing = self.store.load_open_interest(symbol, timeframe)
        fetch_start = self._next_start(
            existing,
            requested_start=start,
            step_ms=self.client.timeframe_ms(timeframe),
        )
        self._log_sync_plan(symbol, f"{timeframe}-open-interest", existing, fetch_start, end)
        if fetch_start > end:
            logger.info("[%s-%s-open-interest] data is already loaded up to %s", symbol, timeframe, _format_ts(end))
            return 0
        frame = self.client.fetch_open_interest(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
        written = self.store.save_open_interest(symbol, timeframe, frame)
        logger.info("%s open interest %s: %s new points", symbol, timeframe, written)
        return written

    @staticmethod
    def _next_start(frame: pd.DataFrame, *, requested_start: pd.Timestamp, step_ms: int) -> pd.Timestamp:
        if frame.empty:
            return requested_start
        latest = pd.to_datetime(frame["timestamp"], errors="coerce").max()
        if pd.isna(latest):
            return requested_start
        latest = _normalize_timestamp(latest)
        return max(requested_start, latest + pd.to_timedelta(step_ms, unit="ms"))

    @staticmethod
    def _log_sync_plan(
        symbol: str,
        timeframe: str,
        existing: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        last_ts = _last_timestamp(existing)
        logger.info(
            "[%s-%s] sync plan: last_ts=%s, start_ts=%s, end_ts=%s",
            symbol,
            timeframe,
            _format_ts(last_ts),
            _format_ts(start),
            _format_ts(end),
        )


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.to_datetime(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _last_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "timestamp" not in frame.columns:
        return None
    latest = pd.to_datetime(frame["timestamp"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return _normalize_timestamp(latest)


def _format_ts(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None:
        return "None"
    return pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
