from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.pipeline.runtime_market_cache import MarketContext
from src_refactor.core.contracts import MarketBatchStream
from src_refactor.core.types import MarketDataBatch
from src_refactor.domain.trading import TradingEngineStepResult


@dataclass(frozen=True, slots=True)
class RuntimeLoopResult:
    steps: tuple[PipelineStepResult, ...] = ()
    final_result: TradingEngineStepResult | None = None


@dataclass(frozen=True, slots=True)
class RuntimeLoop:
    stream: MarketBatchStream
    pipeline: TradingPipeline
    close_open_positions: bool = False
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    def run(self) -> RuntimeLoopResult:
        steps: list[PipelineStepResult] = []
        last_mark_batch: MarketDataBatch | None = None
        for batch in self.stream.stream():
            if self._batch_inside_mark_window(batch):
                last_mark_batch = batch
            contexts = self.pipeline.contexts_from_batch(batch)
            if not contexts or not self._inside_window(contexts):
                continue
            steps.append(self.pipeline.process_contexts(contexts))

        final_result = None
        if self.close_open_positions and last_mark_batch is not None:
            final_result = self.pipeline.trading_engine.close_all_positions(
                timestamp=last_mark_batch.timestamp,
                mark_prices=last_mark_batch.mark_prices,
                reason="FINAL",
            )
        return RuntimeLoopResult(steps=tuple(steps), final_result=final_result)

    def _batch_inside_mark_window(self, batch: MarketDataBatch) -> bool:
        if self.start is not None and batch.timestamp < pd.to_datetime(self.start):
            return False
        if self.end is not None and batch.timestamp > pd.to_datetime(self.end):
            return False
        return True

    def _inside_window(self, contexts: tuple[MarketContext, ...]) -> bool:
        timestamp = min(context.snapshot.current_timestamp for context in contexts)
        if self.start is not None and timestamp < pd.to_datetime(self.start):
            return False
        if self.end is not None and timestamp > pd.to_datetime(self.end):
            return False
        return True
