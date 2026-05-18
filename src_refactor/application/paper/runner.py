from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeAdapters, RuntimeConfig, build_pipeline_from_runtime
from src_refactor.application.runtime_loop import RuntimeLoop
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import MarketDataSubscription
from src_refactor.infrastructure.feeds import LiveFeedMarketStream


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
        result = await asyncio.to_thread(
            RuntimeLoop(
                stream=LiveFeedMarketStream(self.feed, self.subscriptions),
                pipeline=self.pipeline,
            ).run
        )
        self.steps.extend(result.steps)
        return PaperRunResult(steps=tuple(self.steps))


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
