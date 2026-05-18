from __future__ import annotations

from dataclasses import dataclass

from src_refactor.application.pipeline.idempotency_guard import IdempotencyGuard
from src_refactor.application.pipeline.prediction_source import PredictionSource
from src_refactor.core.types import Candle, MarketDataEvent
from src_refactor.domain.signals import SignalBatchProcessor, SignalCandidate, build_signal_id
from src_refactor.domain.trading import TradingEngine, TradingEngineStepResult
from src_refactor.application.pipeline.runtime_market_cache import MarketContext, RuntimeMarketCache


@dataclass(frozen=True, slots=True)
class PipelineStepResult:
    context: MarketContext
    result: TradingEngineStepResult


@dataclass(slots=True)
class TradingPipeline:
    market_cache: RuntimeMarketCache
    trading_engine: TradingEngine
    prediction_source: PredictionSource
    signal_selector: SignalBatchProcessor
    idempotency_guard: IdempotencyGuard

    def on_event(self, event: MarketDataEvent) -> PipelineStepResult | None:
        return self.on_candle(event.candle)

    def on_candle(self, candle: Candle) -> PipelineStepResult | None:
        context = self.market_cache.update(candle)
        if context is None:
            return None
        return self.on_context(context)

    def on_context(self, context: MarketContext) -> PipelineStepResult:
        candidates = self._fresh_candidates(context)
        result = self.trading_engine.on_market_batch(
            bar_index=context.bar_index,
            snapshots={context.symbol: context.snapshot},
            candidates=candidates,
        )
        return PipelineStepResult(context=context, result=result)

    def _fresh_candidates(self, context: MarketContext) -> list[SignalCandidate]:
        predictions = self.prediction_source.predictions_for(context)
        candidates = self.signal_selector.build_candidates(predictions)
        return [
            candidate
            for candidate in candidates
            if self.idempotency_guard.record_once(build_signal_id(candidate))
        ]
