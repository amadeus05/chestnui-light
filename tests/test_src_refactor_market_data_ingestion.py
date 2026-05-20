from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader
from src_refactor.infrastructure.exchanges.bybit import BybitMarketDataClient
from src_refactor.infrastructure.market_data import MarketDataIngestionService, ParquetMarketDataStore


class FakeExchangeClient:
    exchange_code = "bybit"

    def __init__(self) -> None:
        self.candle_starts: list[pd.Timestamp] = []

    def timeframe_ms(self, timeframe: str) -> int:
        assert timeframe == "1h"
        return 3_600_000

    def fetch_candles(self, *, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.candle_starts.append(pd.to_datetime(start))
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start=start, periods=2, freq="h"),
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [10.0, 20.0],
                "quote_volume": [15.0, 50.0],
            }
        )

    def fetch_premium_index_klines(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": [start], "premium_index_close": [1.0]})

    def fetch_funding_rates(self, *, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": [start], "funding_rate": [0.0001]})

    def fetch_open_interest(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": [start], "open_interest": [1000.0]})


@pytest.fixture
def local_tmp_path():
    path = Path("src_refactor/.tmp_tests") / f"market_data_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_parquet_market_data_store_uses_exchange_raw_layout(local_tmp_path):
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    written = store.save_candles(
        "BTC/USDT",
        "1h",
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
                "open": [1, 9],
                "high": [2, 10],
                "low": [0.5, 8],
                "close": [1.5, 9.5],
                "volume": [10, 90],
            }
        ),
    )

    assert written == 1
    assert store.raw_path("candles", symbol="BTC/USDT", timeframe="1h").as_posix().endswith(
        "exchange=bybit/raw/candles/1h/BTC_USDT.parquet"
    )
    loaded = store.load_candles("BTC/USDT", "1h")
    assert len(loaded) == 1
    assert loaded.iloc[0]["open"] == 9


def test_parquet_market_data_store_saves_labeled_layout(local_tmp_path):
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    written = store.save_labeled(
        "BTC/USDT",
        "1h",
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00"],
                "feature_a": [1.0],
                "target": [1],
            }
        ),
    )

    assert written == 1
    assert store.labeled_path(symbol="BTC/USDT", timeframe="1h").as_posix().endswith(
        "exchange=bybit/labeled/1h/BTC_USDT.parquet"
    )
    loaded = store.load_labeled("BTC/USDT", "1h")
    assert loaded.iloc[0]["target"] == 1


def test_market_data_ingestion_writes_incremental_parquet(local_tmp_path):
    client = FakeExchangeClient()
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    service = MarketDataIngestionService(client=client, store=store)

    first = service.backfill_raw(
        symbols=("BTC/USDT",),
        timeframes=("1h",),
        start="2024-01-01 00:00:00",
        end="2024-01-01 03:00:00",
    )
    second = service.backfill_raw(
        symbols=("BTC/USDT",),
        timeframes=("1h",),
        start="2024-01-01 00:00:00",
        end="2024-01-01 03:00:00",
        include_premium_index=False,
        include_funding=False,
        include_open_interest=False,
    )

    assert first.total_written_rows == 5
    assert second.written_rows["candles:1h:BTC/USDT"] == 2
    assert client.candle_starts == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 02:00:00"),
    ]


def test_market_data_ingestion_accepts_mixed_timezone_bounds(local_tmp_path):
    client = FakeExchangeClient()
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    service = MarketDataIngestionService(client=client, store=store)

    summary = service.backfill_raw(
        symbols=("BTC/USDT",),
        timeframes=("1h",),
        start="2024-01-01",
        end=pd.Timestamp.now(tz="UTC"),
        include_premium_index=False,
        include_funding=False,
        include_open_interest=False,
    )

    assert summary.written_rows["candles:1h:BTC/USDT"] == 2
    assert client.candle_starts == [pd.Timestamp("2024-01-01 00:00:00")]


def test_market_data_ingestion_logs_legacy_style_summary(local_tmp_path, caplog):
    client = FakeExchangeClient()
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    service = MarketDataIngestionService(client=client, store=store)

    with caplog.at_level(logging.INFO):
        service.backfill_raw(
            symbols=("BTC/USDT",),
            timeframes=("1h",),
            start="2024-01-01 00:00:00",
            end="2024-01-01 03:00:00",
            include_premium_index=False,
            include_funding=False,
            include_open_interest=False,
        )

    assert "[BTC/USDT-1h] sync plan: last_ts=None" in caplog.text
    assert "BTC/USDT 1h: 2 new candles" in caplog.text


def test_bybit_market_data_client_logs_legacy_style_window_progress(monkeypatch, caplog):
    client = BybitMarketDataClient()

    def fake_get(self, path, params):
        return {
            "retCode": 0,
            "result": {
                "list": [
                    ["1735689600000", "1", "2", "0.5", "1.5", "10", "15"],
                ]
            },
        }

    monkeypatch.setattr(BybitMarketDataClient, "_get", fake_get)

    with caplog.at_level(logging.INFO):
        client.fetch_candles(
            symbol="BTC/USDT",
            timeframe="1h",
            start=pd.Timestamp("2025-01-01 00:00:00"),
            end=pd.Timestamp("2025-01-01 00:00:00"),
        )

    assert "[BTC/USDT-1h] Bybit backfill: 1 windows, limit=1000, workers=1" in caplog.text
    assert "[BTC/USDT-1h] windows 1/1, up to" in caplog.text


def test_bybit_market_data_client_parses_string_millisecond_timestamps(monkeypatch):
    client = BybitMarketDataClient()

    def fake_get(self, path, params):
        return {
            "retCode": 0,
            "result": {
                "list": [
                    ["1735689600000", "1", "2", "0.5", "1.5", "10", "15"],
                ]
            },
        }

    monkeypatch.setattr(BybitMarketDataClient, "_get", fake_get)

    frame = client.fetch_candles(
        symbol="BTC/USDT",
        timeframe="1h",
        start=pd.Timestamp("2025-01-01 00:00:00"),
        end=pd.Timestamp("2025-01-01 00:00:00"),
    )

    assert len(frame) == 1
    assert frame.iloc[0]["timestamp"] == pd.Timestamp("2025-01-01 00:00:00")


def test_parquet_market_data_store_is_candle_repository(local_tmp_path):
    store = ParquetMarketDataStore(root=local_tmp_path, exchange_code="bybit")
    store.save_candles(
        "ETH/USDT",
        "1h",
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00"],
                "open": [1],
                "high": [2],
                "low": [0.5],
                "close": [1.5],
                "volume": [10],
            }
        ),
    )

    frame = HistoricalCandleFrameLoader(store).load_symbol("ETH/USDT", "1h")

    assert frame.iloc[0]["symbol"] == "ETH/USDT"
    assert frame.iloc[0]["timeframe"] == "1h"
