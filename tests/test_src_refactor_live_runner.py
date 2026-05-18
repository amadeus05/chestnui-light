import asyncio
from collections.abc import Callable

import pandas as pd

from src_refactor.application.live import build_live_runner
from src_refactor.application.pipeline import InMemoryIdempotencyGuard, RuntimeMarketCache, StoredPredictionSource
from src_refactor.application.runtime_builder import RuntimeAdapters, RuntimeConfig, TradingMode
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import Candle, MarketDataEvent, MarketDataSubscription, ModelSpec
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
    runner = build_live_runner(
        config=RuntimeConfig(
            mode=TradingMode.LIVE,
            model=ModelSpec(model_type="lightgbm", timeframe="1h"),
        ),
        runtime=runtime,
        subscriptions=subscriptions,
    )

    result = asyncio.run(runner.run())

    assert feed.subscriptions == subscriptions
    assert feed.closed is True
    assert len(result.steps) == 1
    assert result.steps[0].result.opened_orders == ()


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
