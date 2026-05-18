import pandas as pd

from src_refactor.application.pipeline import StoredPredictionSource, TradingPipeline
from src_refactor.application.pipeline.idempotency_guard import InMemoryIdempotencyGuard
from src_refactor.application.pipeline.runtime_market_cache import RuntimeMarketCache
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode, build_backtest_runtime
from src_refactor.core.types import ModelSpec, Prediction
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.domain.trading import TradingEngine
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator


def _prediction(symbol: str, p_long: float, timestamp: str = "2025-01-01 00:00:00") -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp(timestamp),
        symbol=symbol,
        timeframe="1h",
        model_id="test_model",
        direction=0,
        confidence=p_long,
        proba_long=p_long,
        proba_short=1.0 - p_long,
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


def test_pipeline_batches_all_symbols_before_selecting_best_signal():
    market_cache = RuntimeMarketCache()
    contexts = market_cache.update_from_frame(_market_frame())
    pipeline = TradingPipeline(
        market_cache=market_cache,
        trading_engine=TradingEngine(
            broker=ExchangeSimulator(),
            portfolio=PortfolioManager(initial_balance=100.0),
            risk=RiskManager(),
        ),
        prediction_source=StoredPredictionSource.from_predictions(
            [
                _prediction("BTC/USDT", 0.70),
                _prediction("ETH/USDT", 0.90),
            ]
        ),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
    )

    result = pipeline.on_contexts(contexts)

    assert len(result.result.opened_orders) == 1
    assert result.result.opened_orders[0].symbol == "ETH/USDT"


def test_new_position_can_exit_on_entry_candle():
    market_cache = RuntimeMarketCache()
    contexts = market_cache.update_from_frame(
        pd.DataFrame(
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
    )
    portfolio = PortfolioManager(initial_balance=100.0)
    pipeline = TradingPipeline(
        market_cache=market_cache,
        trading_engine=TradingEngine(
            broker=ExchangeSimulator(),
            portfolio=portfolio,
            risk=RiskManager(),
        ),
        prediction_source=StoredPredictionSource.from_predictions(
            [_prediction("BTC/USDT", 0.90, timestamp="2025-01-01 00:00:00")]
        ),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
    )

    result = pipeline.on_contexts(contexts)

    assert len(result.result.opened_orders) == 1
    assert len(result.result.closed_trades) == 1
    assert result.result.closed_trades[0].reason == "TP"
    assert portfolio.position_snapshot("BTC/USDT") is None


def test_runtime_config_can_use_legacy_trading_parameters_as_single_source():
    class LegacyConfig:
        BACKTEST_INITIAL_BALANCE = 250.0
        TAKER_COM = 0.001
        SLIPPAGE = 0.002
        RISK_PER_TRADE = 0.03
        LEVERAGE = 4.0
        BACKTEST_MAX_OPEN_POSITIONS = 3
        BACKTEST_SL_COOLDOWN_BARS = 8
        BACKTEST_MAX_SL_PER_DAY = 2
        BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = 2
        BACKTEST_REDUCED_RISK_PER_TRADE = 0.01
        DIRECTIONAL_PROBA_THRESHOLD = 0.62
        MIN_SIGNAL_GAP = 0.04
        ALLOW_LONGS = False
        ALLOW_SHORTS = True
        BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 2

    runtime_config = RuntimeConfig.from_legacy_config(
        mode=TradingMode.BACKTEST,
        model=ModelSpec(model_type="lightgbm", timeframe="1h"),
        config=LegacyConfig,
    )
    adapters = build_backtest_runtime(
        runtime_config,
        prediction_source=StoredPredictionSource.from_predictions([]),
    )

    assert runtime_config.initial_balance == 250.0
    assert runtime_config.pricing.taker_fee == 0.001
    assert runtime_config.pricing.slippage == 0.002
    assert runtime_config.risk.risk_per_trade == 0.03
    assert runtime_config.risk.leverage == 4.0
    assert runtime_config.risk.max_open_positions == 3
    assert runtime_config.risk.sl_cooldown_bars == 8
    assert runtime_config.risk.max_sl_per_day == 2
    assert runtime_config.risk.reduce_risk_after_consecutive_losses == 2
    assert runtime_config.risk.reduced_risk_per_trade == 0.01
    assert runtime_config.signals.directional_proba_threshold == 0.62
    assert runtime_config.signals.min_signal_gap == 0.04
    assert runtime_config.signals.allow_longs is False
    assert runtime_config.signals.allow_shorts is True
    assert runtime_config.trading.max_new_positions_per_bar == 2
    assert adapters.signal_selector.config == runtime_config.signals
    assert adapters.broker.pricing == runtime_config.pricing
