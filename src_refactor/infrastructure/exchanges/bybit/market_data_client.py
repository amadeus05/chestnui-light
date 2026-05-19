from __future__ import annotations

from dataclasses import dataclass
from time import sleep

import pandas as pd
import requests


BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

BYBIT_OPEN_INTEREST_INTERVALS = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


@dataclass(slots=True)
class BybitMarketDataClient:
    base_url: str = "https://api.bybit.com"
    category: str = "linear"
    limit: int = 1000
    funding_limit: int = 200
    open_interest_limit: int = 200
    timeout: float = 20.0
    retry_count: int = 5
    retry_sleep: float = 0.3

    @property
    def exchange_code(self) -> str:
        return "bybit"

    def timeframe_ms(self, timeframe: str) -> int:
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported Bybit timeframe: {timeframe}")
        return TIMEFRAME_MS[timeframe]

    def fetch_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        interval = self._interval(timeframe)
        rows: list[list[object]] = []
        for window_start, window_end in self._windows(start, end, self.timeframe_ms(timeframe), self.limit):
            payload = self._get(
                "/v5/market/kline",
                {
                    "category": self.category,
                    "symbol": _api_symbol(symbol),
                    "interval": interval,
                    "start": window_start,
                    "end": window_end,
                    "limit": self.limit,
                },
            )
            rows.extend(payload.get("result", {}).get("list", []))
        return _kline_frame(rows)

    def fetch_premium_index_klines(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        interval = self._interval(timeframe)
        rows: list[list[object]] = []
        for window_start, window_end in self._windows(start, end, self.timeframe_ms(timeframe), self.limit):
            payload = self._get(
                "/v5/market/premium-index-price-kline",
                {
                    "category": self.category,
                    "symbol": _api_symbol(symbol),
                    "interval": interval,
                    "start": window_start,
                    "end": window_end,
                    "limit": self.limit,
                },
            )
            rows.extend(payload.get("result", {}).get("list", []))
        return _premium_index_frame(rows)

    def fetch_funding_rates(
        self,
        *,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        interval_ms = 8 * 60 * 60 * 1000
        rows: list[dict[str, object]] = []
        for window_start, window_end in self._windows(start, end, interval_ms, self.funding_limit):
            payload = self._get(
                "/v5/market/funding/history",
                {
                    "category": self.category,
                    "symbol": _api_symbol(symbol),
                    "startTime": window_start,
                    "endTime": window_end,
                    "limit": self.funding_limit,
                },
            )
            rows.extend(payload.get("result", {}).get("list", []))
        return _funding_frame(rows)

    def fetch_open_interest(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        if timeframe not in BYBIT_OPEN_INTEREST_INTERVALS:
            raise ValueError(f"Unsupported Bybit open interest timeframe: {timeframe}")
        rows: list[dict[str, object]] = []
        for window_start, window_end in self._windows(
            start,
            end,
            self.timeframe_ms(timeframe),
            self.open_interest_limit,
        ):
            payload = self._get(
                "/v5/market/open-interest",
                {
                    "category": self.category,
                    "symbol": _api_symbol(symbol),
                    "intervalTime": BYBIT_OPEN_INTEREST_INTERVALS[timeframe],
                    "startTime": window_start,
                    "endTime": window_end,
                    "limit": self.open_interest_limit,
                },
            )
            rows.extend(payload.get("result", {}).get("list", []))
        return _open_interest_frame(rows)

    def _interval(self, timeframe: str) -> str:
        if timeframe not in BYBIT_INTERVALS:
            raise ValueError(f"Unsupported Bybit timeframe: {timeframe}")
        return BYBIT_INTERVALS[timeframe]

    def _windows(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe_ms: int,
        limit: int,
    ) -> list[tuple[int, int]]:
        start_ms = _timestamp_ms(start)
        end_ms = _timestamp_ms(end)
        if start_ms > end_ms:
            return []

        max_span_ms = timeframe_ms * max(int(limit) - 1, 1)
        windows: list[tuple[int, int]] = []
        window_start = start_ms
        while window_start <= end_ms:
            window_end = min(window_start + max_span_ms, end_ms)
            windows.append((window_start, window_end))
            window_start = window_end + timeframe_ms
        return windows

    def _get(self, path: str, params: dict[str, object]) -> dict:
        last_error: Exception | None = None
        url = f"{self.base_url}{path}"
        for attempt in range(max(1, self.retry_count)):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") == 0:
                    return payload
                raise RuntimeError(f"Bybit retCode={payload.get('retCode')}, retMsg={payload.get('retMsg')}")
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < self.retry_count:
                    sleep(self.retry_sleep * (2 ** attempt))
        raise RuntimeError(f"Bybit request failed: {last_error}")


def _kline_frame(rows: list[list[object]]) -> pd.DataFrame:
    frame = _rows_to_frame(
        rows,
        columns=("timestamp", "open", "high", "low", "close", "volume", "quote_volume"),
    )
    return _normalize_numeric_frame(frame, ("open", "high", "low", "close", "volume", "quote_volume"))


def _premium_index_frame(rows: list[list[object]]) -> pd.DataFrame:
    frame = _rows_to_frame(
        [row[:5] for row in rows if len(row) >= 5],
        columns=("timestamp", "open", "high", "low", "premium_index_close"),
    )
    return _normalize_numeric_frame(frame, ("open", "high", "low", "premium_index_close"))


def _funding_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "timestamp": row.get("fundingRateTimestamp"),
                "funding_rate": row.get("fundingRate"),
            }
            for row in rows
        ]
    )
    return _normalize_numeric_frame(frame, ("funding_rate",))


def _open_interest_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "timestamp": row.get("timestamp"),
                "open_interest": row.get("openInterest"),
            }
            for row in rows
        ]
    )
    return _normalize_numeric_frame(frame, ("open_interest",))


def _rows_to_frame(rows: list[list[object]], *, columns: tuple[str, ...]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _normalize_numeric_frame(frame: pd.DataFrame, numeric_columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], unit="ms", errors="coerce")
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values(
        "timestamp"
    ).reset_index(drop=True)


def _api_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def _timestamp_ms(value: pd.Timestamp) -> int:
    timestamp = pd.to_datetime(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)
