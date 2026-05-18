import pandas as pd

from src_refactor.application.pipeline import InMemoryIdempotencyGuard, RuntimeMarketCache, StoredPredictionSource, TradingPipeline
from src_refactor.application.runtime_loop import RuntimeLoop
from src_refactor.core.types import Prediction
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.domain.trading import TradingEngine
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator
from src_refactor.infrastructure.feeds import HistoricalFrameMarketStream


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01 01:00:00"),
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
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 1_000.0,
            },
        ]
    )


def _prediction(symbol: str, p_long: float) -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        symbol=symbol,
        timeframe="1h",
        model_id="model",
        direction=0,
        confidence=p_long,
        proba_long=p_long,
        proba_short=1.0 - p_long,
        raw={"barrier_stop_pct": 0.02, "barrier_take_pct": 0.04},
    )


def test_historical_frame_stream_yields_one_batch_per_timestamp():
    batches = list(HistoricalFrameMarketStream(_frame()).stream())

    assert [batch.timestamp for batch in batches] == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 01:00:00"),
    ]
    assert set(batches[0].by_symbol) == {"BTC/USDT", "ETH/USDT"}


def test_runtime_loop_processes_historical_stream_as_symbol_batches():
    pipeline = TradingPipeline(
        market_cache=RuntimeMarketCache(),
        trading_engine=TradingEngine(
            broker=ExchangeSimulator(),
            portfolio=PortfolioManager(initial_balance=100.0),
            risk=RiskManager(),
        ),
        prediction_source=StoredPredictionSource.from_predictions(
            [_prediction("BTC/USDT", 0.70), _prediction("ETH/USDT", 0.90)]
        ),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
    )

    result = RuntimeLoop(
        stream=HistoricalFrameMarketStream(_frame()),
        pipeline=pipeline,
    ).run()

    assert len(result.steps) == 1
    assert len(result.steps[0].result.opened_orders) == 1
    assert result.steps[0].result.opened_orders[0].symbol == "ETH/USDT"
