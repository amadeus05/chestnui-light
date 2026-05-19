import pandas as pd

from src_refactor.core.types import MarketDataset


def test_market_dataset_filters_symbols_with_period_overlap():
    dataset = MarketDataset(
        frame=pd.DataFrame(
            [
                _row("2025-01-01 00:00:00", "BTC/USDT"),
                _row("2025-01-01 01:00:00", "BTC/USDT"),
                _row("2025-01-01 00:00:00", "ETH/USDT"),
                _row("2024-12-31 23:00:00", "SOL/USDT"),
                _row("2025-01-01 00:00:00", "SOL/USDT"),
                _row("2025-01-01 01:00:00", "SOL/USDT"),
            ]
        ),
        timeframe="1h",
        symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"),
    )

    filtered = dataset.drop_symbols_with_insufficient_rows(
        start=pd.Timestamp("2025-01-01 00:00:00"),
        end=pd.Timestamp("2025-01-01 01:00:00"),
        min_rows=2,
    )

    assert filtered.symbols == ("BTC/USDT", "SOL/USDT")
    assert set(filtered.frame["symbol"]) == {"BTC/USDT", "SOL/USDT"}


def test_market_dataset_between_keeps_symbols_metadata_unchanged():
    dataset = MarketDataset(
        frame=pd.DataFrame(
            [
                _row("2024-12-31 23:00:00", "BTC/USDT"),
                _row("2025-01-01 00:00:00", "BTC/USDT"),
            ]
        ),
        timeframe="1h",
        symbols=("BTC/USDT",),
    )

    filtered = dataset.between(
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 01:00:00"),
    )

    assert filtered.symbols == ("BTC/USDT",)
    assert filtered.frame["timestamp"].tolist() == [pd.Timestamp("2025-01-01 00:00:00")]


def _row(timestamp: str, symbol: str) -> dict[str, object]:
    return {
        "timestamp": pd.Timestamp(timestamp),
        "symbol": symbol,
        "timeframe": "1h",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1_000.0,
    }
