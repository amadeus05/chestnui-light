import asyncio
from collections.abc import Callable

import pandas as pd

from src_refactor.application.live import LiveRunner
from src_refactor.application.pipeline import InMemoryIdempotencyGuard, RuntimeMarketCache, StoredPredictionSource
from src_refactor.application.runtime_builder import (
    RuntimeAdapters,
    RuntimeConfig,
    TradingMode,
    build_paper_runtime,
    build_runtime,
)
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelInputBuilder, ModelPredictor
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import (
    Candle,
    LightGbmInput,
    MarketDataEvent,
    MarketDataSubscription,
    ModelInput,
    ModelSpec,
    Prediction,
)
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator
from src_refactor.infrastructure.feeds import LiveFeedMarketStream


class FakeLiveFeed(LiveMarketDataFeed):
    def __init__(self, events: list[MarketDataEvent]) -> None:
        self.events = events
        self.subscriptions: list[MarketDataSubscription] = []
        self.closed = False

    async def subscribe(self, subscriptions: list[MarketDataSubscription]) -> None:
        self.subscriptions = subscriptions

    async def run(self, on_event: Callable[[MarketDataEvent], None]) -> None:
        for event in self.events:
            on_event(event)

    async def close(self) -> None:
        self.closed = True


def _event(timestamp: str, symbol: str = "BTC/USDT") -> MarketDataEvent:
    return MarketDataEvent(
        candle=Candle(
            symbol=symbol,
            timeframe="1h",
            timestamp=pd.Timestamp(timestamp),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000.0,
        )
    )


def test_live_runner_uses_same_pipeline_path_as_streaming_runtime():
    feed = FakeLiveFeed([_event("2025-01-01 00:00:00"), _event("2025-01-01 01:00:00")])
    subscriptions = [MarketDataSubscription(symbol="BTC/USDT", timeframe="1h")]
    runtime = RuntimeAdapters(
        market_cache=RuntimeMarketCache(),
        broker=ExchangeSimulator(),
        prediction_source=StoredPredictionSource.from_predictions([]),
        signal_selector=SignalBatchProcessor(),
        idempotency_guard=InMemoryIdempotencyGuard(),
        data_source=feed,
    )
    config = RuntimeConfig(
        mode=TradingMode.LIVE,
        model=ModelSpec(model_type="lightgbm", timeframe="1h"),
    )
    runner = LiveRunner(
        stream=LiveFeedMarketStream(feed, subscriptions),
        pipeline=build_runtime(config, runtime).pipeline,
    )

    result = asyncio.run(runner.run())

    assert feed.subscriptions == subscriptions
    assert feed.closed is True
    assert len(result.steps) == 1
    assert result.steps[0].result.opened_orders == ()


def test_paper_runtime_uses_effective_predictor_model_spec_metadata():
    config = RuntimeConfig(
        mode=TradingMode.PAPER,
        model=ModelSpec(model_type="lightgbm", timeframe="1h"),
    )
    predictor_spec = ModelSpec(
        model_type="lightgbm",
        timeframe="1h",
        metadata={
            "feature_columns": ["ema_fast_slow"],
            "labeling": {"use_dynamic_barriers": False, "sl_pct": 0.02, "tp_pct": 0.04},
        },
    )

    adapters = build_paper_runtime(
        config,
        registry=FakeRegistry(predictor_spec),  # type: ignore[arg-type]
        data_source=FakeLiveFeed([]),
    )

    assert adapters.prediction_source.model_spec is predictor_spec


def test_live_feed_market_stream_batches_subscribed_symbols_by_timestamp():
    feed = FakeLiveFeed(
        [
            _event("2025-01-01 00:00:00", "ETH/USDT"),
            _event("2025-01-01 00:00:00", "BTC/USDT"),
            _event("2025-01-01 01:00:00", "BTC/USDT"),
            _event("2025-01-01 01:00:00", "ETH/USDT"),
        ]
    )
    subscriptions = [
        MarketDataSubscription(symbol="BTC/USDT", timeframe="1h"),
        MarketDataSubscription(symbol="ETH/USDT", timeframe="1h"),
    ]

    batches = list(LiveFeedMarketStream(feed, subscriptions).stream())

    assert feed.subscriptions == subscriptions
    assert feed.closed is True
    assert [batch.timestamp for batch in batches] == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 01:00:00"),
    ]
    assert [set(batch.by_symbol) for batch in batches] == [
        {"BTC/USDT", "ETH/USDT"},
        {"BTC/USDT", "ETH/USDT"},
    ]


class NoopInputBuilder(ModelInputBuilder):
    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> ModelInput:
        raise NotImplementedError

    def build_predict_input(self, request, spec=None) -> ModelInput:
        return LightGbmInput(features=pd.DataFrame({"x": [1.0]}), feature_names=("x",))


class SpecPredictor(ModelPredictor):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def predict(self, model_input: ModelInput) -> Prediction:
        return Prediction(
            timestamp=pd.Timestamp("2025-01-01"),
            symbol="BTC/USDT",
            timeframe=self.spec.timeframe,
            model_id=self.spec.model_id,
            direction=0,
            confidence=0.8,
        )


class FakeBundle:
    def __init__(self, predictor_spec: ModelSpec) -> None:
        self.input_builder = NoopInputBuilder()
        self.predictor = SpecPredictor(predictor_spec)

    def load_predictor(self, fold_id: int | None = None) -> ModelPredictor:
        return self.predictor


class FakeRegistry:
    def __init__(self, predictor_spec: ModelSpec) -> None:
        self.bundle = FakeBundle(predictor_spec)

    def get(self, spec: ModelSpec) -> FakeBundle:
        return self.bundle
