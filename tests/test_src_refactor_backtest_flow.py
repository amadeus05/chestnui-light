import pandas as pd

from src_refactor.application.backtest import StoredPredictionBacktestFlow, StoredPredictionBacktestRequest
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.core.types import ModelSpec, Prediction
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
