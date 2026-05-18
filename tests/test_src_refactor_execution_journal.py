import pandas as pd

from src_refactor.application.pipeline import InMemoryIdempotencyGuard, RuntimeMarketCache, StoredPredictionSource, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode, build_backtest_runtime, build_runtime
from src_refactor.core.types import ModelSpec, Prediction
from src_refactor.domain.execution import ExecutionJournal
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.domain.trading import TradingEngine
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator


def _prediction() -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        symbol="BTC/USDT",
        timeframe="1h",
        model_id="model",
        direction=0,
        confidence=0.9,
        proba_long=0.9,
        proba_short=0.1,
        raw={"barrier_stop_pct": 0.02, "barrier_take_pct": 0.04},
    )


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01 01:00:00"),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open": 100.0,
                "high": 105.0,
                "low": 99.5,
                "close": 104.0,
                "volume": 1_000.0,
            },
        ]
    )


def test_execution_journal_records_orders_fills_trades_and_account_snapshots():
    journal = ExecutionJournal()
    market_cache = RuntimeMarketCache()
    contexts = market_cache.update_from_frame(_market_frame())
    pipeline = TradingPipeline(
        market_cache=market_cache,
        trading_engine=TradingEngine(
            broker=ExchangeSimulator(),
            portfolio=PortfolioManager(initial_balance=100.0),
            risk=RiskManager(),
            execution_journal=journal,
        ),
        prediction_source=StoredPredictionSource.from_predictions([_prediction()]),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
    )

    result = pipeline.on_contexts(contexts)

    event_types = [event.event_type for event in journal.events]
    assert event_types == [
        "ORDER_SUBMITTED",
        "ORDER_ACCEPTED",
        "FILL",
        "FILL",
        "TRADE_CLOSED",
        "ACCOUNT_SNAPSHOT",
    ]
    assert [event.sequence for event in journal.events] == list(range(1, len(journal.events) + 1))
    assert result.result.accepted_orders[0].status == "open"
    assert result.result.account is not None
    assert result.result.account.positions == {}
    assert journal.events[-1].account == result.result.account


def test_backtest_runtime_does_not_wire_execution_journal():
    journal = ExecutionJournal()
    config = RuntimeConfig(
        mode=TradingMode.BACKTEST,
        model=ModelSpec(model_type="lightgbm", timeframe="1h"),
    )
    adapters = build_backtest_runtime(
        config,
        prediction_source=StoredPredictionSource.from_predictions([_prediction()]),
    )
    runtime = build_runtime(config, adapters)
    contexts = runtime.pipeline.market_cache.update_from_frame(_market_frame())

    runtime.pipeline.on_contexts(contexts)

    assert runtime.pipeline.trading_engine.execution_journal is None
    assert journal.events == []
