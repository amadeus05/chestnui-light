from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src_refactor.application.pipeline import (
    IdempotencyGuard,
    InMemoryIdempotencyGuard,
    ModelPredictionSource,
    PredictionSource,
    RuntimeMarketCache,
    TradingPipeline,
)
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.contracts.historical_market_feed import HistoricalMarketFeed
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import ModelSpec
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.domain.trading import TradingEngine, TradingEngineConfig
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator
from src_refactor.infrastructure.models import ModelRegistry

MarketDataSource = HistoricalMarketFeed | LiveMarketDataFeed


class TradingMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: TradingMode
    model: ModelSpec
    initial_balance: float = 100.0
    fold_id: int | None = None
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeAdapters:
    market_cache: RuntimeMarketCache
    broker: BrokerGateway
    prediction_source: PredictionSource
    signal_selector: SignalBatchProcessor
    idempotency_guard: IdempotencyGuard
    data_source: MarketDataSource | None = None


@dataclass(frozen=True, slots=True)
class Runtime:
    config: RuntimeConfig
    adapters: RuntimeAdapters
    pipeline: TradingPipeline


def build_runtime(config: RuntimeConfig, runtime_adapters: RuntimeAdapters) -> Runtime:
    return Runtime(
        config=config,
        adapters=runtime_adapters,
        pipeline=build_pipeline_from_runtime(config, runtime_adapters),
    )


def build_pipeline_from_runtime(config: RuntimeConfig, runtime: RuntimeAdapters) -> TradingPipeline:
    return TradingPipeline(
        market_cache=runtime.market_cache,
        trading_engine=TradingEngine(
            broker=runtime.broker,
            portfolio=PortfolioManager(initial_balance=config.initial_balance),
            risk=RiskManager(),
            config=TradingEngineConfig(),
        ),
        prediction_source=runtime.prediction_source,
        signal_selector=runtime.signal_selector,
        idempotency_guard=runtime.idempotency_guard,
    )


def build_backtest_runtime(
    config: RuntimeConfig,
    *,
    prediction_source: PredictionSource,
    data_source: HistoricalMarketFeed | None = None,
    broker: BrokerGateway | None = None,
    market_cache: RuntimeMarketCache | None = None,
) -> RuntimeAdapters:
    return RuntimeAdapters(
        market_cache=market_cache or RuntimeMarketCache(),
        broker=broker or ExchangeSimulator(),
        prediction_source=prediction_source,
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
        data_source=data_source,
    )


def build_paper_runtime(
    config: RuntimeConfig,
    *,
    registry: ModelRegistry,
    data_source: LiveMarketDataFeed,
    broker: BrokerGateway | None = None,
    market_cache: RuntimeMarketCache | None = None,
) -> RuntimeAdapters:
    bundle = registry.get(config.model)
    return RuntimeAdapters(
        market_cache=market_cache or RuntimeMarketCache(),
        broker=broker or ExchangeSimulator(),
        prediction_source=ModelPredictionSource(
            input_builder=bundle.input_builder,
            predictor=bundle.load_predictor(fold_id=config.fold_id),
            model_spec=config.model,
        ),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
        data_source=data_source,
    )


def build_live_runtime(
    config: RuntimeConfig,
    *,
    registry: ModelRegistry,
    data_source: LiveMarketDataFeed,
    broker: BrokerGateway,
    market_cache: RuntimeMarketCache | None = None,
) -> RuntimeAdapters:
    bundle = registry.get(config.model)
    return RuntimeAdapters(
        market_cache=market_cache or RuntimeMarketCache(),
        broker=broker,
        prediction_source=ModelPredictionSource(
            input_builder=bundle.input_builder,
            predictor=bundle.load_predictor(fold_id=config.fold_id),
            model_spec=config.model,
        ),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
        data_source=data_source,
    )
