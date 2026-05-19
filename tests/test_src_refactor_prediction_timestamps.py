import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src_refactor.application.pipeline import StoredPredictionSource
from src_refactor.application.pipeline.runtime_market_cache import RuntimeMarketCache
from src_refactor.core.types import Prediction
from src_refactor.infrastructure.predictions import InMemoryPredictionStore, ParquetPredictionStore
from src_refactor.infrastructure.predictions.timestamps import canonical_prediction_timestamp


def _prediction(timestamp: object, symbol: str = "BTC/USDT") -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp(timestamp),
        symbol=symbol,
        timeframe="1h",
        model_id="model",
        direction=0,
        confidence=0.7,
        proba_long=0.7,
        proba_short=0.3,
        signal_gap=0.4,
        stop_pct=0.02,
        take_pct=0.04,
    )


def _context_for(timestamp: str):
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(timestamp),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01 01:00:00"),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "volume": 1_000.0,
            },
        ]
    )
    return RuntimeMarketCache().update_from_frame(frame)[-1]


def test_canonical_prediction_timestamp_matches_naive_and_utc_aware_values():
    naive = pd.Timestamp("2025-01-01 00:00:00")
    aware = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    assert canonical_prediction_timestamp(naive) == canonical_prediction_timestamp(aware)
    assert canonical_prediction_timestamp(aware).tzinfo is None


def test_stored_prediction_source_matches_utc_prediction_to_naive_context():
    source = StoredPredictionSource.from_predictions(
        [_prediction(pd.Timestamp("2025-01-01 00:00:00", tz="UTC"))]
    )

    predictions = source.predictions_for(_context_for("2025-01-01 00:00:00"))

    assert len(predictions) == 1
    assert predictions[0].symbol == "BTC/USDT"


def test_in_memory_prediction_store_filters_naive_and_aware_ranges():
    store = InMemoryPredictionStore()
    store.write(
        [
            _prediction(pd.Timestamp("2025-01-01 00:00:00", tz="UTC")),
            _prediction(pd.Timestamp("2025-01-02 00:00:00", tz="UTC"), symbol="ETH/USDT"),
        ]
    )

    predictions = store.read(
        model_id="model",
        start=pd.Timestamp("2025-01-01 00:00:00"),
        end=pd.Timestamp("2025-01-01 23:59:59"),
    )

    assert [prediction.symbol for prediction in predictions] == ["BTC/USDT"]


def test_stored_prediction_source_reads_from_prediction_store():
    store = InMemoryPredictionStore()
    store.write(
        [
            _prediction("2025-01-01 00:00:00", symbol="BTC/USDT"),
            _prediction("2025-01-02 00:00:00", symbol="ETH/USDT"),
            Prediction(
                timestamp=pd.Timestamp("2025-01-01 00:00:00"),
                symbol="SOL/USDT",
                timeframe="1h",
                model_id="other",
                direction=0,
                confidence=0.8,
            ),
        ]
    )

    source = StoredPredictionSource.from_store(
        store,
        model_id="model",
        symbols=("BTC/USDT",),
        start=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        end=pd.Timestamp("2025-01-01 00:00:00"),
    )

    predictions = source.predictions_for(_context_for("2025-01-01 00:00:00"))

    assert [prediction.symbol for prediction in predictions] == ["BTC/USDT"]


def test_parquet_prediction_store_round_trips_canonical_timestamps():
    path = Path("src_refactor/.tmp_tests") / f"predictions_{uuid4().hex}.parquet"
    try:
        store = ParquetPredictionStore(path)
        store.write([_prediction(pd.Timestamp("2025-01-01 00:00:00", tz="UTC"))])

        predictions = store.read(
            model_id="model",
            start=pd.Timestamp("2025-01-01 00:00:00"),
            end=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        )

        assert len(predictions) == 1
        assert predictions[0].timestamp == pd.Timestamp("2025-01-01 00:00:00")
        assert predictions[0].timestamp.tzinfo is None
        assert predictions[0].signal_gap == pytest.approx(0.4)
        assert predictions[0].stop_pct == pytest.approx(0.02)
        assert predictions[0].take_pct == pytest.approx(0.04)
    finally:
        path.unlink(missing_ok=True)


def test_parquet_prediction_store_reads_legacy_raw_barriers():
    path = Path("src_refactor/.tmp_tests") / f"legacy_predictions_{uuid4().hex}.parquet"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "model_id": "model",
                    "direction": 0,
                    "confidence": 0.7,
                    "fold_id": None,
                    "proba_long": 0.7,
                    "proba_short": 0.3,
                    "raw_json": json.dumps(
                        {
                            "signal_gap": 0.4,
                            "barrier_stop_pct": 0.02,
                            "barrier_take_pct": 0.04,
                        }
                    ),
                }
            ]
        ).to_parquet(path, index=False)

        predictions = ParquetPredictionStore(path).read(model_id="model")

        assert len(predictions) == 1
        assert predictions[0].signal_gap == pytest.approx(0.4)
        assert predictions[0].stop_pct == pytest.approx(0.02)
        assert predictions[0].take_pct == pytest.approx(0.04)
    finally:
        path.unlink(missing_ok=True)
