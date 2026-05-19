import pandas as pd
import pytest
from pathlib import Path

import bt

from src_refactor.application.backtest import StoredPredictionBacktestFlow, StoredPredictionBacktestRequest
from src_refactor.application.training.training_runner import FoldTrainingResult, WalkForwardTrainingResult
from src_refactor.core.config import ExperimentConfig
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.core.types import ModelArtifact, ModelSpec, Prediction, WalkForwardFold
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader
from src_refactor.infrastructure.predictions import InMemoryPredictionStore


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


def test_stored_prediction_backtest_flow_matches_legacy_oos_backtest(monkeypatch, capsys):
    symbol = "BTC/USDT"
    model = ModelSpec(model_type="lightgbm", timeframe="1h")
    predictions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "symbol": symbol,
                "p_short": 0.1,
                "p_long": 0.9,
            }
        ]
    )
    barriers = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "symbol": symbol,
                "barrier_stop_pct": 0.02,
                "barrier_take_pct": 0.04,
            }
        ]
    )
    market_frame = FakeCandleRepository().load_candles(symbol, "1h")
    normalized_market = market_frame.assign(
        timestamp=pd.to_datetime(market_frame["timestamp"], unit="ms"),
        symbol=symbol,
        timeframe="1h",
    )

    chart_path = Path("src_refactor/.tmp_tests/legacy_parity_equity.png")
    try:
        _patch_legacy_backtest(monkeypatch, normalized_market, barriers, symbol)
        bt.backtest(
            features_meta={
                "feature_columns": [],
                "train_period": {
                    "start": "2025-01-01 00:00:00",
                    "end": "2025-01-01 01:00:00",
                },
                "symbols": [symbol],
            },
            predictions=predictions,
            equity_curve_path=chart_path,
            result_title="LEGACY TEST",
        )
    finally:
        chart_path.unlink(missing_ok=True)
    legacy_stdout = capsys.readouterr().out

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

    assert "Trades: 1" in legacy_stdout
    assert "TP" in legacy_stdout
    assert result.metrics is not None
    assert result.metrics.summary.total_trades == 1
    assert closed_trade.reason == "TP"
    assert closed_trade.pnl_abs == pytest.approx(_legacy_expected_tp_pnl_abs())


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


class FakeCandleRepository:
    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        assert symbol == "BTC/USDT"
        assert timeframe == "1h"
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


def _patch_legacy_backtest(monkeypatch, market_frame: pd.DataFrame, barriers: pd.DataFrame, symbol: str) -> None:
    monkeypatch.setattr(bt, "SYMBOLS", [symbol], raising=False)
    monkeypatch.setattr(bt, "TIMEFRAME", "1h", raising=False)
    monkeypatch.setattr(bt, "HTF_TIMEFRAME", "4h", raising=False)
    monkeypatch.setattr(bt, "ALLOW_LONGS", True, raising=False)
    monkeypatch.setattr(bt, "ALLOW_SHORTS", True, raising=False)
    monkeypatch.setattr(bt, "DIRECTIONAL_PROBA_THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(bt, "MIN_SIGNAL_GAP", 0.0, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_INITIAL_BALANCE", 100.0, raising=False)
    monkeypatch.setattr(bt, "RISK_PER_TRADE", 0.01, raising=False)
    monkeypatch.setattr(bt, "LEVERAGE", 1.0, raising=False)
    monkeypatch.setattr(bt, "TAKER_COM", 0.0004, raising=False)
    monkeypatch.setattr(bt, "SLIPPAGE", 0.0003, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_MAX_NEW_POSITIONS_PER_BAR", 1, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_MAX_OPEN_POSITIONS", 1, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_SL_COOLDOWN_BARS", 0, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_MAX_SL_PER_DAY", 0, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0, raising=False)
    monkeypatch.setattr(bt, "BACKTEST_REDUCED_RISK_PER_TRADE", 0.01, raising=False)
    monkeypatch.setattr(bt.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "axhline", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(bt.plt, "show", lambda: None)

    legacy_frame = market_frame.copy()
    legacy_frame["close_time"] = legacy_frame["timestamp"] + pd.to_timedelta(1, unit="h")
    legacy_features = barriers.copy()

    monkeypatch.setattr(
        bt,
        "load_all_raw_data",
        lambda symbols: {
            symbol: {
                "main": legacy_frame.copy(),
                "htf": legacy_frame.copy(),
            }
        },
    )
    monkeypatch.setattr(bt, "load_precomputed_features", lambda *args, **kwargs: legacy_features.copy())


def _legacy_expected_tp_pnl_abs() -> float:
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
