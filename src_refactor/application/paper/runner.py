from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.runtime_loop import RuntimeLoop
from src_refactor.core.contracts import MarketBatchStream


@dataclass(frozen=True, slots=True)
class PaperRunResult:
    steps: tuple[PipelineStepResult, ...] = ()


@dataclass(slots=True)
class PaperRunner:
    stream: MarketBatchStream
    pipeline: TradingPipeline
    steps: list[PipelineStepResult] = field(default_factory=list)

    async def run(self) -> PaperRunResult:
        result = await asyncio.to_thread(
            RuntimeLoop(
                stream=self.stream,
                pipeline=self.pipeline,
            ).run
        )
        self.steps.extend(result.steps)
        return PaperRunResult(steps=tuple(self.steps))
