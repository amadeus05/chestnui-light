from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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
from src_refactor.domain.execution import ExecutionJournal, ExecutionPricingConfig
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskConfig, RiskManager
from src_refactor.domain.signals import SignalBatchProcessor, SignalProcessingConfig
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
    pricing: ExecutionPricingConfig = field(default_factory=ExecutionPricingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signals: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)
    trading: TradingEngineConfig = field(default_factory=TradingEngineConfig)

    @classmethod
    def from_legacy_config(
        cls,
        *,
        mode: TradingMode,
        model: ModelSpec,
        config: Any,
        fold_id: int | None = None,
        symbols: tuple[str, ...] = (),
    ) -> "RuntimeConfig":
        return cls(
            mode=mode,
            model=model,
            initial_balance=float(getattr(config, "BACKTEST_INITIAL_BALANCE", 100.0)),
            fold_id=fold_id,
            symbols=symbols,
            pricing=ExecutionPricingConfig.from_legacy_config(config),
            risk=RiskConfig(
                risk_per_trade=float(getattr(config, "RISK_PER_TRADE", 0.01)),
                leverage=float(getattr(config, "LEVERAGE", 1.0)),
                min_position_notional=float(getattr(config, "MIN_POSITION_NOTIONAL", 10.0)),
                max_open_positions=int(getattr(config, "BACKTEST_MAX_OPEN_POSITIONS", 1)),
                sl_cooldown_bars=int(getattr(config, "BACKTEST_SL_COOLDOWN_BARS", 0)),
                max_sl_per_day=int(getattr(config, "BACKTEST_MAX_SL_PER_DAY", 0)),
                reduce_risk_after_consecutive_losses=int(
                    getattr(config, "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0)
                ),
                reduced_risk_per_trade=float(getattr(config, "BACKTEST_REDUCED_RISK_PER_TRADE"))
                if getattr(config, "BACKTEST_REDUCED_RISK_PER_TRADE", None) is not None
                else None,
            ),
            signals=SignalProcessingConfig(
                directional_proba_threshold=float(
                    getattr(config, "DIRECTIONAL_PROBA_THRESHOLD", getattr(config, "CONFIDENCE_THRESHOLD", 0.5))
                ),
                min_signal_gap=float(getattr(config, "MIN_SIGNAL_GAP", 0.0)),
                allow_longs=bool(getattr(config, "ALLOW_LONGS", True)),
                allow_shorts=bool(getattr(config, "ALLOW_SHORTS", True)),
            ),
            trading=TradingEngineConfig(
                max_new_positions_per_bar=int(getattr(config, "BACKTEST_MAX_NEW_POSITIONS_PER_BAR", 1))
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeAdapters:
    market_cache: RuntimeMarketCache
    broker: BrokerGateway
    prediction_source: PredictionSource
    signal_selector: SignalBatchProcessor
    idempotency_guard: IdempotencyGuard
    data_source: MarketDataSource | None = None
    execution_journal: ExecutionJournal | None = None


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
            portfolio=PortfolioManager(initial_balance=config.initial_balance, pricing=config.pricing),
            risk=RiskManager(config.risk),
            config=config.trading,
            execution_journal=runtime.execution_journal,
        ),
        prediction_source=runtime.prediction_source,
        signal_selector=runtime.signal_selector,
        idempotency_guard=runtime.idempotency_guard,
    )


def build_runtime_adapters(
    config: RuntimeConfig,
    *,
    prediction_source: PredictionSource | None = None,
    registry: ModelRegistry | None = None,
    data_source: MarketDataSource | None = None,
    broker: BrokerGateway | None = None,
    market_cache: RuntimeMarketCache | None = None,
    execution_journal: ExecutionJournal | None = None,
) -> RuntimeAdapters:
    if prediction_source is None:
        if registry is None:
            raise ValueError("Runtime adapters require either prediction_source or registry.")
        bundle = registry.get(config.model)
        prediction_source = build_model_prediction_source(config, bundle)

    if broker is None:
        if config.mode == TradingMode.LIVE:
            raise ValueError("Live runtime requires an explicit broker.")
        broker = ExchangeSimulator(pricing=config.pricing)

    return RuntimeAdapters(
        market_cache=market_cache or RuntimeMarketCache(),
        broker=broker,
        prediction_source=prediction_source,
        signal_selector=SignalBatchProcessor(config.signals),
        idempotency_guard=InMemoryIdempotencyGuard(),
        data_source=data_source,
        execution_journal=None if config.mode == TradingMode.BACKTEST else execution_journal,
    )


def build_model_prediction_source(config: RuntimeConfig, bundle: Any) -> ModelPredictionSource:
    predictor = bundle.load_predictor(fold_id=config.fold_id)
    return ModelPredictionSource(
        input_builder=bundle.input_builder,
        predictor=predictor,
        model_spec=_effective_model_spec(config.model, predictor),
    )


def _effective_model_spec(default: ModelSpec, predictor: Any) -> ModelSpec:
    predictor_spec = getattr(predictor, "spec", None)
    return predictor_spec if isinstance(predictor_spec, ModelSpec) else default
