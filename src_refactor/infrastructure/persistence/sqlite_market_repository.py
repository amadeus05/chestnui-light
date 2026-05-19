from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class SqliteMarketRepository:
    db_path: str | Path
    exchange_code: str = "bybit"

    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_market_frame(
            f"""
            SELECT open_time AS timestamp, open, high, low, close, volume
            FROM {self._candles_table_name()}
            WHERE symbol=? AND timeframe=?
            ORDER BY open_time
            """,
            (symbol, timeframe),
            numeric_columns=("open", "high", "low", "close", "volume"),
        )

    def load_funding_rates(self, symbol: str) -> pd.DataFrame:
        return self._read_market_frame(
            f"""
            SELECT funding_time AS timestamp, funding_rate
            FROM {self._funding_table_name()}
            WHERE symbol=?
            ORDER BY funding_time
            """,
            (symbol,),
            numeric_columns=("funding_rate",),
        )

    def load_premium_index_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_market_frame(
            f"""
            SELECT open_time AS timestamp, close AS premium_index_close
            FROM {self._premium_index_table_name()}
            WHERE symbol=? AND timeframe=?
            ORDER BY open_time
            """,
            (symbol, timeframe),
            numeric_columns=("premium_index_close",),
        )

    def load_open_interest(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._read_market_frame(
            f"""
            SELECT timestamp, open_interest
            FROM {self._open_interest_table_name()}
            WHERE symbol=? AND timeframe=?
            ORDER BY timestamp
            """,
            (symbol, timeframe),
            numeric_columns=("open_interest",),
        )

    def _read_market_frame(
        self,
        query: str,
        params: tuple[object, ...],
        *,
        numeric_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        db_file = Path(self.db_path)
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")

        with sqlite3.connect(db_file) as connection:
            frame = pd.read_sql_query(query, connection, params=params)
        if frame.empty:
            return frame

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["timestamp"]).reset_index(drop=True)

    def _candles_table_name(self) -> str:
        return "candles" if self.exchange_code == "bybit" else f"candles_{self.exchange_code}"

    def _funding_table_name(self) -> str:
        return "funding_rates" if self.exchange_code == "bybit" else f"funding_rates_{self.exchange_code}"

    def _premium_index_table_name(self) -> str:
        return "premium_index_klines" if self.exchange_code == "bybit" else f"premium_index_klines_{self.exchange_code}"

    def _open_interest_table_name(self) -> str:
        return "open_interest" if self.exchange_code == "bybit" else f"open_interest_{self.exchange_code}"
