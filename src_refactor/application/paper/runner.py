from __future__ import annotations

from dataclasses import dataclass, field

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeAdapters, RuntimeConfig, build_pipeline_from_runtime
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import MarketDataEvent, MarketDataSubscription


@dataclass(frozen=True, slots=True)
class PaperRunResult:
    steps: tuple[PipelineStepResult, ...] = ()


@dataclass(slots=True)
class PaperRunner:
    feed: LiveMarketDataFeed
    subscriptions: list[MarketDataSubscription]
    pipeline: TradingPipeline
    steps: list[PipelineStepResult] = field(default_factory=list)

    async def run(self) -> PaperRunResult:
        await self.feed.subscribe(self.subscriptions)
        await self.feed.run(self.on_event)
        return PaperRunResult(steps=tuple(self.steps))

    def on_event(self, event: MarketDataEvent) -> None:
        result = self.pipeline.on_event(event)
        if result is not None:
            self.steps.append(result)


def build_paper_runner(
    *,
    config: RuntimeConfig,
    runtime: RuntimeAdapters,
    subscriptions: list[MarketDataSubscription],
) -> PaperRunner:
    if not isinstance(runtime.data_source, LiveMarketDataFeed):
        raise ValueError("Paper runtime requires LiveMarketDataFeed data_source.")
    return PaperRunner(
        feed=runtime.data_source,
        subscriptions=subscriptions,
        pipeline=build_pipeline_from_runtime(config, runtime),
    )
