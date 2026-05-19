import pandas as pd
import pytest
from pathlib import Path
from uuid import uuid4

from src_refactor.application.backtest import StoredPredictionBacktestFlow, StoredPredictionBacktestRequest
from src_refactor.application.backtest import walk_forward_oos_flow as wvf_backtest_module
from src_refactor.application.backtest import WalkForwardOosBacktestFlow
from src_refactor.application.training.training_runner import FoldTrainingResult, WalkForwardTrainingResult
from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig
from src_refactor.core.config import ExperimentConfig
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.core.types import ModelArtifact, ModelSpec, Prediction, WalkForwardFold
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader, HistoricalMarketMapLoader
from src_refactor.infrastructure.predictions import InMemoryPredictionStore, ParquetPredictionStore


def test_stored_prediction_backtest_flow_loads_data_and_runs_runtime():
    model = ModelSpec(model_type="lightgbm", timeframe="1h")
    store = InMemoryPredictionStore()
    store.write(
        [
            Prediction(
                timestamp=pd.Timestamp("2025-01-01 00:00:00"),
                symbol="BTC/USDT",
                timeframe="1h",
                model_id=model.model_id,
                direction=0,
                confidence=0.9,
                proba_long=0.9,
                proba_short=0.1,
                signal_gap=0.8,
                stop_pct=0.02,
                take_pct=0.04,
            )
        ]
    )
    flow = StoredPredictionBacktestFlow(
        config=RuntimeConfig(
            mode=TradingMode.BACKTEST,
            model=model,
            symbols=("BTC/USDT",),
        ),
        candle_loader=HistoricalCandleFrameLoader(FakeCandleRepository()),
        prediction_store=store,
    )

    result = flow.run(StoredPredictionBacktestRequest())

    assert len(result.steps) == 1
    assert len(result.steps[0].result.opened_orders) == 1
    assert len(result.steps[0].result.closed_trades) == 1
    assert result.steps[0].result.closed_trades[0].reason == "TP"
    assert result.metrics is not None
    assert result.metrics.summary.total_trades == 1


def test_stored_prediction_backtest_flow_matches_expected_oos_execution_math():
    symbol = "BTC/USDT"
    model = ModelSpec(model_type="lightgbm", timeframe="1h")
    store = InMemoryPredictionStore()
    store.write(
        [
            Prediction(
                timestamp=pd.Timestamp("2025-01-01 00:00:00"),
                symbol=symbol,
                timeframe="1h",
                model_id=model.model_id,
                direction=0,
                confidence=0.9,
                proba_long=0.9,
                proba_short=0.1,
                signal_gap=0.8,
                stop_pct=0.02,
                take_pct=0.04,
            )
        ]
    )
    flow = StoredPredictionBacktestFlow(
        config=RuntimeConfig(
            mode=TradingMode.BACKTEST,
            model=model,
            symbols=(symbol,),
        ),
        candle_loader=HistoricalCandleFrameLoader(FakeCandleRepository()),
        prediction_store=store,
    )

    result = flow.run(StoredPredictionBacktestRequest())
    closed_trade = result.steps[0].result.closed_trades[0]

    assert result.metrics is not None
    assert result.metrics.summary.total_trades == 1
    assert closed_trade.reason == "TP"
    assert closed_trade.pnl_abs == pytest.approx(_expected_tp_pnl_abs())


def test_stored_prediction_backtest_flow_requires_symbols():
    flow = StoredPredictionBacktestFlow(
        config=RuntimeConfig(
            mode=TradingMode.BACKTEST,
            model=ModelSpec(model_type="lightgbm", timeframe="1h"),
        ),
        candle_loader=HistoricalCandleFrameLoader(FakeCandleRepository()),
        prediction_store=InMemoryPredictionStore(),
    )

    try:
        flow.run()
    except ValueError as exc:
        assert "requires symbols" in str(exc)
    else:
        raise AssertionError("Expected StoredPredictionBacktestFlow to require symbols.")


def test_stored_prediction_backtest_request_uses_training_result_window():
    model = ModelSpec(model_type="lightgbm", timeframe="1h", symbols=("BTC/USDT",))
    training_result = WalkForwardTrainingResult(
        config=ExperimentConfig(model=model, symbols=("BTC/USDT",)),
        prediction_store_path=Path("models/predictions/oos.parquet"),
        folds=[
            FoldTrainingResult(
                fold=WalkForwardFold(
                    fold_id=0,
                    train_start=pd.Timestamp("2024-01-01"),
                    train_end=pd.Timestamp("2024-12-31"),
                    test_start=pd.Timestamp("2025-01-01 00:00:00"),
                    test_end=pd.Timestamp("2025-01-01 01:00:00"),
                ),
                artifact=ModelArtifact(spec=model, uri="model.joblib", fold_id=0),
                predictions=[
                    _prediction_for_request("2025-01-01 01:00:00", model),
                    _prediction_for_request("2025-01-01 00:00:00", model),
                ],
            )
        ],
    )

    request = StoredPredictionBacktestRequest.from_training_result(training_result)

    assert request.symbols == ("BTC/USDT",)
    assert request.timeframe == "1h"
    assert request.model_id == model.model_id
    assert request.start == pd.Timestamp("2025-01-01 00:00:00")
    assert request.end == pd.Timestamp("2025-01-01 01:00:00")
    assert training_result.prediction_store_path == Path("models/predictions/oos.parquet")


def test_stored_prediction_backtest_flow_builds_from_training_result_store_path():
    model = ModelSpec(model_type="lightgbm", timeframe="1h", symbols=("BTC/USDT",))
    training_result = WalkForwardTrainingResult(
        config=ExperimentConfig(model=model, symbols=("BTC/USDT",)),
        prediction_store_path=Path("models/predictions/oos.parquet"),
        folds=[],
    )

    flow = StoredPredictionBacktestFlow.from_training_result(
        config=RuntimeConfig(mode=TradingMode.BACKTEST, model=model, symbols=("BTC/USDT",)),
        candle_loader=HistoricalCandleFrameLoader(FakeCandleRepository()),
        result=training_result,
    )

    assert isinstance(flow.prediction_store, ParquetPredictionStore)
    assert flow.prediction_store.path == Path("models/predictions/oos.parquet")


def test_stored_prediction_backtest_flow_requires_training_prediction_store_path():
    model = ModelSpec(model_type="lightgbm", timeframe="1h", symbols=("BTC/USDT",))
    training_result = WalkForwardTrainingResult(
        config=ExperimentConfig(model=model, symbols=("BTC/USDT",)),
        folds=[],
    )

    with pytest.raises(ValueError, match="prediction_store_path"):
        StoredPredictionBacktestFlow.from_training_result(
            config=RuntimeConfig(mode=TradingMode.BACKTEST, model=model, symbols=("BTC/USDT",)),
            candle_loader=HistoricalCandleFrameLoader(FakeCandleRepository()),
            result=training_result,
        )


def test_walk_forward_oos_backtest_flow_routes_predictions_through_runtime(monkeypatch):
    symbol = "BTC/USDT"
    model = ModelSpec(model_type="lightgbm", timeframe="1h", symbols=(symbol,))
    predictions_dir = Path("src_refactor/.tmp_tests")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"oos_{uuid4().hex}.parquet"

    def fake_run_wvf_oos(*, base_candle_map, htf_candle_map, config):
        assert set(base_candle_map) == {symbol}
        assert set(htf_candle_map) == {symbol}
        prediction = _prediction_for_request("2025-01-01 00:00:00", model)
        prediction = Prediction(
            timestamp=prediction.timestamp,
            symbol=prediction.symbol,
            timeframe=prediction.timeframe,
            model_id=prediction.model_id,
            direction=1,
            confidence=0.9,
            proba_long=0.9,
            proba_short=0.1,
            signal_gap=0.8,
            stop_pct=0.02,
            take_pct=0.04,
        )
        ParquetPredictionStore(config.predictions_path).write([prediction])
        return WalkForwardTrainingResult(
            config=ExperimentConfig(model=model, symbols=(symbol,)),
            prediction_store_path=config.predictions_path,
            folds=[
                FoldTrainingResult(
                    fold=WalkForwardFold(
                        fold_id=0,
                        train_start=pd.Timestamp("2024-01-01"),
                        train_end=pd.Timestamp("2024-12-31"),
                        test_start=pd.Timestamp("2025-01-01 00:00:00"),
                        test_end=pd.Timestamp("2025-01-01 00:00:00"),
                    ),
                    artifact=ModelArtifact(spec=model, uri="model.joblib", fold_id=0),
                    predictions=[prediction],
                )
            ],
        )

    monkeypatch.setattr(wvf_backtest_module, "run_wvf_oos", fake_run_wvf_oos)

    result = WalkForwardOosBacktestFlow(
        market_loader=HistoricalMarketMapLoader.from_repository(FakeCandleRepository()),
        htf_timeframe="4h",
    ).run(
        training_config=WvfOosRunConfig(
            symbols=(symbol,),
            timeframe="1h",
            predictions_path=predictions_path,
        ),
        runtime_config=RuntimeConfig(mode=TradingMode.BACKTEST, model=model, symbols=(symbol,)),
    )

    assert result.training.prediction_store_path == predictions_path
    assert result.backtest.metrics is not None
    assert result.backtest.metrics.summary.total_trades == 1


def test_historical_market_map_loader_attaches_derivative_context():
    frame = HistoricalMarketMapLoader.from_repository(FakeMarketRepository()).load_base_map(("BTC/USDT",), "1h")[
        "BTC/USDT"
    ]

    assert frame["funding_rate"].tolist() == [0.001, 0.001]
    assert frame["premium_index_close"].tolist() == [1.01, 1.01]
    assert frame["open_interest"].tolist() == [10_000.0, 10_000.0]


class FakeCandleRepository:
    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        assert symbol == "BTC/USDT"
        assert timeframe in {"1h", "4h"}
        return pd.DataFrame(
            [
                {
                    "timestamp": 1735689600000,
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100.0,
                    "volume": 1_000.0,
                },
                {
                    "timestamp": 1735693200000,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.5,
                    "close": 104.0,
                    "volume": 1_000.0,
                },
            ]
        )


class FakeMarketRepository(FakeCandleRepository):
    def load_funding_rates(self, symbol: str) -> pd.DataFrame:
        assert symbol == "BTC/USDT"
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-12-31 23:00:00")],
                "funding_rate": [0.001],
            }
        )

    def load_premium_index_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        assert symbol == "BTC/USDT"
        assert timeframe == "1h"
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-12-31 23:00:00")],
                "premium_index_close": [1.01],
            }
        )

    def load_open_interest(self, symbol: str, timeframe: str) -> pd.DataFrame:
        assert symbol == "BTC/USDT"
        assert timeframe == "1h"
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-12-31 23:00:00")],
                "open_interest": [10_000.0],
            }
        )


def _expected_tp_pnl_abs() -> float:
    entry_price = 100.0 * (1 + 0.0003)
    exit_price = (entry_price * (1 + 0.04)) * (1 - 0.0003)
    pnl_pct = (exit_price - entry_price) / entry_price - (0.0004 + 0.0004)
    position_notional = min(100.0 * 0.01 / 0.02, 100.0 * 1.0)
    return position_notional * pnl_pct


def _prediction_for_request(timestamp: str, model: ModelSpec) -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp(timestamp),
        symbol="BTC/USDT",
        timeframe=model.timeframe,
        model_id=model.model_id,
        direction=0,
        confidence=0.9,
    )
