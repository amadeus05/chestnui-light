from __future__ import annotations

from dataclasses import dataclass, field

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeAdapters, RuntimeConfig, build_pipeline_from_runtime
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import MarketDataBatch, MarketDataEvent, MarketDataSubscription


@dataclass(frozen=True, slots=True)
class LiveRunResult:
    steps: tuple[PipelineStepResult, ...] = ()


@dataclass(slots=True)
class LiveRunner:
    feed: LiveMarketDataFeed
    subscriptions: list[MarketDataSubscription]
    pipeline: TradingPipeline
    steps: list[PipelineStepResult] = field(default_factory=list)

    async def run(self) -> LiveRunResult:
        await self.feed.subscribe(self.subscriptions)
        try:
            await self.feed.run(self.on_event)
        finally:
            await self.feed.close()
        return LiveRunResult(steps=tuple(self.steps))

    def on_event(self, event: MarketDataEvent) -> None:
        result = self.pipeline.process_batch(MarketDataBatch.from_candles([event.candle]))
        if result is not None:
            self.steps.append(result)


def build_live_runner(
    *,
    config: RuntimeConfig,
    runtime: RuntimeAdapters,
    subscriptions: list[MarketDataSubscription],
) -> LiveRunner:
    if not isinstance(runtime.data_source, LiveMarketDataFeed):
        raise ValueError("Live runtime requires LiveMarketDataFeed data_source.")
    return LiveRunner(
        feed=runtime.data_source,
        subscriptions=subscriptions,
        pipeline=build_pipeline_from_runtime(config, runtime),
    )
