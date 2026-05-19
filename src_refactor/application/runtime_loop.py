from __future__ import annotations

from dataclasses import dataclass

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.core.contracts import MarketBatchStream
from src_refactor.domain.trading import TradingEngineStepResult


@dataclass(frozen=True, slots=True)
class RuntimeLoopResult:
    steps: tuple[PipelineStepResult, ...] = ()
    final_result: TradingEngineStepResult | None = None


@dataclass(frozen=True, slots=True)
class RuntimeLoop:
    stream: MarketBatchStream
    pipeline: TradingPipeline

    def run(self) -> RuntimeLoopResult:
        steps: list[PipelineStepResult] = []
        for batch in self.stream.stream():
            contexts = self.pipeline.contexts_from_batch(batch)
            if not contexts:
                continue
            steps.append(self.pipeline.process_contexts(contexts))

        return RuntimeLoopResult(steps=tuple(steps))
