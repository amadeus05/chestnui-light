from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "premium_index_close",
    "funding_rate",
    "open_interest",
}


@dataclass(frozen=True, slots=True)
class ParquetMarketDataStore:
    root: str | Path = "_data"
    exchange_code: str = "bybit"

    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_raw_frame("candles", symbol=symbol, timeframe=timeframe)

    def save_candles(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        return self._merge_raw_frame("candles", frame, symbol=symbol, timeframe=timeframe)

    def load_premium_index_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_raw_frame("premium_index", symbol=symbol, timeframe=timeframe)

    def save_premium_index_klines(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        return self._merge_raw_frame("premium_index", frame, symbol=symbol, timeframe=timeframe)

    def load_funding_rates(self, symbol: str) -> pd.DataFrame:
        return self._read_raw_frame("funding", symbol=symbol, timeframe=None)

    def save_funding_rates(self, symbol: str, frame: pd.DataFrame) -> int:
        return self._merge_raw_frame("funding", frame, symbol=symbol, timeframe=None)

    def load_open_interest(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_raw_frame("open_interest", symbol=symbol, timeframe=timeframe)

    def save_open_interest(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        return self._merge_raw_frame("open_interest", frame, symbol=symbol, timeframe=timeframe)

    def raw_path(self, dataset: str, *, symbol: str, timeframe: str | None = None) -> Path:
        parts = [
            Path(self.root),
            f"exchange={self.exchange_code}",
            "raw",
            dataset,
        ]
        if timeframe is not None:
            parts.append(timeframe)
        return Path(*parts) / f"{_symbol_key(symbol)}.parquet"

    def labeled_path(self, *, symbol: str, timeframe: str) -> Path:
        return (
            Path(self.root)
            / f"exchange={self.exchange_code}"
            / "labeled"
            / timeframe
            / f"{_symbol_key(symbol)}.parquet"
        )

    def load_labeled(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.labeled_path(symbol=symbol, timeframe=timeframe)
        if not path.exists():
            return pd.DataFrame()
        return _normalize_frame(pd.read_parquet(path))

    def save_labeled(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        prepared = _normalize_frame(frame)
        if prepared.empty:
            return 0

        path = self.labeled_path(symbol=symbol, timeframe=timeframe)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        before = len(existing)
        merged = _dedupe_by_timestamp(pd.concat([existing, prepared], ignore_index=True))

        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        return max(0, len(merged) - before)

    def _read_raw_frame(self, dataset: str, *, symbol: str, timeframe: str | None) -> pd.DataFrame:
        path = self.raw_path(dataset, symbol=symbol, timeframe=timeframe)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        return _normalize_frame(frame)

    def _merge_raw_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str | None,
    ) -> int:
        prepared = _normalize_frame(frame)
        if prepared.empty:
            return 0

        path = self.raw_path(dataset, symbol=symbol, timeframe=timeframe)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        before = len(existing)
        merged = _dedupe_by_timestamp(pd.concat([existing, prepared], ignore_index=True))

        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        return max(0, len(merged) - before)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "timestamp" not in frame.columns:
        raise ValueError("Market data frame must contain timestamp column.")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    output = output.dropna(subset=["timestamp"])
    for column in output.columns:
        if column in NUMERIC_COLUMNS:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.sort_values("timestamp").reset_index(drop=True)


def _dedupe_by_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = _normalize_frame(frame)
    return output.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def _symbol_key(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "_")
