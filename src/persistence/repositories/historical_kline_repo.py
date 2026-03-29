from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

import config as cfg
from config import DB_PATH
from src.contracts.exchange_contract import ExchangeContract
from src.persistence.sqlite_connection import create_sqlite_connection
from src.types.common import HistoricalKline, Symbol


logger = logging.getLogger(__name__)


class HistoricalKlineRepository:
    LEGACY_EXCHANGE_CODE = "bybit"

    def __init__(
        self,
        db_path: str = DB_PATH,
        exchange_code: str | None = None,
        connection_factory: Callable[[], sqlite3.Connection] | None = None,
    ) -> None:
        self.db_path = db_path
        self.exchange_code = self._normalize_exchange_code(exchange_code or getattr(cfg, "ACTIVE_EXCHANGE", "bybit"))
        self.connection_factory = connection_factory or (lambda: create_sqlite_connection(self.db_path))

    def init_schema(self) -> None:
        candles_table = self._candles_table_name()
        sync_state_table = self._sync_state_table_name()
        with self.connection_factory() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {candles_table} (
                    symbol TEXT,
                    timeframe TEXT,
                    open_time INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    quote_volume REAL,
                    PRIMARY KEY (symbol, timeframe, open_time)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {sync_state_table} (
                    dataset TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    empty_since_ts INTEGER,
                    last_checked_ts INTEGER,
                    PRIMARY KEY (dataset, symbol, timeframe)
                )
                """
            )
            conn.commit()

    def get_last_open_time(self, symbol: str | Symbol, timeframe: str) -> int | None:
        with self.connection_factory() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT MAX(open_time) FROM {self._candles_table_name()} WHERE symbol=? AND timeframe=?",
                (self._symbol_name(symbol), timeframe),
            )
            return cur.fetchone()[0]

    def get_candle_count(self, symbol: str | Symbol, timeframe: str) -> int:
        with self.connection_factory() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*) FROM {self._candles_table_name()} WHERE symbol=? AND timeframe=?",
                (self._symbol_name(symbol), timeframe),
            )
            return int(cur.fetchone()[0] or 0)

    def get_sync_state(self, dataset: str, symbol: str | Symbol, timeframe: str) -> dict | None:
        with self.connection_factory() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT empty_since_ts, last_checked_ts
                FROM {self._sync_state_table_name()}
                WHERE dataset=? AND symbol=? AND timeframe=?
                """,
                (dataset, self._symbol_name(symbol), timeframe),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "empty_since_ts": row[0],
                "last_checked_ts": row[1],
            }

    def upsert_sync_state(
        self,
        dataset: str,
        symbol: str | Symbol,
        timeframe: str,
        empty_since_ts: int,
        last_checked_ts: int,
    ) -> None:
        with self.connection_factory() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._sync_state_table_name()} (dataset, symbol, timeframe, empty_since_ts, last_checked_ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset, symbol, timeframe) DO UPDATE SET
                    empty_since_ts = excluded.empty_since_ts,
                    last_checked_ts = excluded.last_checked_ts
                """,
                (
                    dataset,
                    self._symbol_name(symbol),
                    timeframe,
                    empty_since_ts,
                    last_checked_ts,
                ),
            )
            conn.commit()

    def clear_sync_state(self, dataset: str, symbol: str | Symbol, timeframe: str) -> None:
        with self.connection_factory() as conn:
            conn.execute(
                f"DELETE FROM {self._sync_state_table_name()} WHERE dataset=? AND symbol=? AND timeframe=?",
                (dataset, self._symbol_name(symbol), timeframe),
            )
            conn.commit()

    def save_candles(
        self,
        symbol: str | Symbol,
        timeframe: str,
        candles: list[HistoricalKline],
    ) -> int:
        if not candles:
            return 0

        rows = [
            (
                self._symbol_name(symbol),
                timeframe,
                candle.open_time,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.quote_volume,
            )
            for candle in candles
        ]
        rows.sort(key=lambda row: row[2])

        with self.connection_factory() as conn:
            before_changes = conn.total_changes
            conn.executemany(
                f"INSERT OR IGNORE INTO {self._candles_table_name()} VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            return conn.total_changes - before_changes

    def load_candles(self, symbol: str | Symbol, timeframe: str) -> pd.DataFrame:
        with self.connection_factory() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT open_time AS timestamp, open, high, low, close, volume
                FROM {self._candles_table_name()}
                WHERE symbol=? AND timeframe=?
                ORDER BY open_time
                """,
                conn,
                params=(self._symbol_name(symbol), timeframe),
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def save_features(self, symbol: str | Symbol, df: pd.DataFrame) -> None:
        table_name = self.feature_table_name(symbol)
        with self.connection_factory() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"{self._symbol_name(symbol)} features saved ({len(df)} rows)")

    def list_tables(self) -> set[str]:
        with self.connection_factory() as conn:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return {row[0] for row in conn.execute(query).fetchall()}

    def load_features(self, symbol: str | Symbol) -> pd.DataFrame:
        table_name = self.feature_table_name(symbol)
        available_tables = self.list_tables()
        symbol_name = self._symbol_name(symbol)
        if table_name not in available_tables:
            logger.warning("Skipping %s: table %s not found in DB", symbol_name, table_name)
            return pd.DataFrame()

        with self.connection_factory() as conn:
            frame = pd.read_sql_query(
                f'SELECT * FROM "{table_name}" ORDER BY "timestamp"',
                conn,
            )
        if frame.empty:
            logger.warning("Skipping %s: table %s is empty", symbol_name, table_name)
            return frame

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["symbol"] = symbol_name
        return frame

    def load_feature_dataset(self, symbols: list[str | Symbol]) -> pd.DataFrame:
        db_file = Path(self.db_path)
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")

        frames = [self.load_features(symbol) for symbol in symbols]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise RuntimeError("No feature tables found. Run etl.py first to populate *_features tables.")

        return pd.concat(frames, ignore_index=True)

    def feature_table_exists(self, symbol: str | Symbol) -> bool:
        return self.feature_table_name(symbol) in self.list_tables()

    def sync_candles(
        self,
        exchange: ExchangeContract,
        symbol: str | Symbol,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        dataset: str = "candles",
    ) -> int:
        exchange_code = self._normalize_exchange_code(exchange.get_exchange_code())
        if exchange_code != self.exchange_code:
            raise ValueError(
                f"Repository exchange_code={self.exchange_code} does not match exchange={exchange_code}"
            )

        normalized_symbol = exchange.normalize_symbol(symbol)
        timeframe_ms = exchange.get_timeframe_ms(timeframe)
        existing_row_count = self.get_candle_count(normalized_symbol, timeframe)
        last_ts = self.get_last_open_time(normalized_symbol, timeframe)

        if (
            existing_row_count == 0
            and last_ts is None
            and self.feature_table_exists(normalized_symbol)
            and not bool(getattr(cfg, "ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY", False))
        ):
            logger.warning(
                f"[{normalized_symbol}-{timeframe}] raw candles are missing, but the feature table already exists. "
                "Skipping automatic full backfill to avoid unexpected re-download. "
                "Set ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY=True to force a rebuild."
            )
            return 0

        if last_ts:
            start_ts = last_ts + timeframe_ms
        else:
            start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)

        end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000) if end_date else int(datetime.now().timestamp() * 1000)
        logger.info(
            f"[{normalized_symbol}-{timeframe}] sync plan: "
            f"rows_in_db={existing_row_count}, "
            f"last_ts={self._format_ts(last_ts)}, "
            f"start_ts={self._format_ts(start_ts)}, "
            f"end_ts={self._format_ts(end_ts)}"
        )
        sync_state = self.get_sync_state(dataset=dataset, symbol=normalized_symbol, timeframe=timeframe)
        if sync_state and sync_state.get("empty_since_ts") is not None:
            empty_since_ts = int(sync_state["empty_since_ts"])
            last_checked_ts = int(sync_state.get("last_checked_ts") or end_ts)
            start_ts = max(start_ts, last_checked_ts + timeframe_ms)
            next_retry_ts = last_checked_ts + timeframe_ms
            if start_ts >= empty_since_ts and end_ts < next_retry_ts:
                logger.info(
                    f"[{normalized_symbol}-{timeframe}] no newer candles after "
                    f"{datetime.fromtimestamp(empty_since_ts / 1000)}; "
                    f"skipping repeated empty backfill until {datetime.fromtimestamp(next_retry_ts / 1000)}"
                )
                return 0

        if start_ts > end_ts:
            logger.info(f"[{normalized_symbol}-{timeframe}] data is already loaded up to {end_date}")
            return 0

        try:
            candles = exchange.fetch_klines(normalized_symbol, timeframe, start_ts, end_ts)
        except Exception as exc:
            logger.error(f"load error for {normalized_symbol}-{timeframe}: {exc}")
            return 0

        total_loaded = self.save_candles(normalized_symbol, timeframe, candles)
        if total_loaded > 0:
            self.clear_sync_state(dataset=dataset, symbol=normalized_symbol, timeframe=timeframe)
            return total_loaded

        empty_since_ts = start_ts
        if candles:
            latest_fetched_ts = max(candle.open_time for candle in candles)
            empty_since_ts = max(empty_since_ts, int(latest_fetched_ts) + timeframe_ms)

        self.upsert_sync_state(
            dataset=dataset,
            symbol=normalized_symbol,
            timeframe=timeframe,
            empty_since_ts=empty_since_ts,
            last_checked_ts=end_ts,
        )
        return total_loaded

    @staticmethod
    def _symbol_name(symbol: str | Symbol) -> str:
        return str(symbol) if isinstance(symbol, Symbol) else str(Symbol.from_string(symbol))

    def feature_table_name(self, symbol: str | Symbol) -> str:
        symbol_stub = self._symbol_name(symbol).replace("/", "_") + "_features"
        if self.exchange_code == self.LEGACY_EXCHANGE_CODE:
            return symbol_stub
        return f"{self.exchange_code}_{symbol_stub}"

    @staticmethod
    def _format_ts(timestamp_ms: int | None) -> str:
        if timestamp_ms is None:
            return "None"
        return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalize_exchange_code(exchange_code: str) -> str:
        return exchange_code.strip().lower()

    def _candles_table_name(self) -> str:
        if self.exchange_code == self.LEGACY_EXCHANGE_CODE:
            return "candles"
        return f"candles_{self.exchange_code}"

    def _sync_state_table_name(self) -> str:
        if self.exchange_code == self.LEGACY_EXCHANGE_CODE:
            return "data_sync_state"
        return f"data_sync_state_{self.exchange_code}"
