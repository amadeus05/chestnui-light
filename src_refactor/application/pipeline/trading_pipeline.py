from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src_refactor.application.pipeline.idempotency_guard import IdempotencyGuard
from src_refactor.application.pipeline.prediction_source import PredictionSource
from src_refactor.core.types import Candle, MarketDataEvent
from src_refactor.domain.signals import SignalBatchProcessor, SignalCandidate, build_signal_id
from src_refactor.domain.trading import TradingEngine, TradingEngineStepResult
from src_refactor.application.pipeline.runtime_market_cache import MarketContext, RuntimeMarketCache


@dataclass(frozen=True, slots=True)
class PipelineStepResult:
    context: MarketContext | tuple[MarketContext, ...]
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
        return self.on_contexts([context])

    def on_contexts(self, contexts: Sequence[MarketContext]) -> PipelineStepResult:
        if not contexts:
            raise ValueError("TradingPipeline.on_contexts requires at least one context.")
        context_tuple = tuple(contexts)
        candidates = self._fresh_candidates(context_tuple)
        result = self.trading_engine.on_market_batch(
            bar_index=min(context.bar_index for context in context_tuple),
            snapshots={context.symbol: context.snapshot for context in context_tuple},
            candidates=candidates,
        )
        context: MarketContext | tuple[MarketContext, ...] = context_tuple[0] if len(context_tuple) == 1 else context_tuple
        return PipelineStepResult(context=context, result=result)

    def _fresh_candidates(self, contexts: Sequence[MarketContext]) -> list[SignalCandidate]:
        snapshot_symbols = {context.symbol for context in contexts}
        predictions = [
            prediction
            for context in contexts
            for prediction in self.prediction_source.predictions_for(context)
            if prediction.symbol in snapshot_symbols
        ]
        candidates = self.signal_selector.build_candidates(predictions)
        fresh_candidates: list[SignalCandidate] = []
        local_seen: set[str] = set()
        for candidate in candidates:
            signal_id = build_signal_id(candidate)
            if signal_id in local_seen:
                continue
            local_seen.add(signal_id)
            if self.idempotency_guard.record_once(signal_id):
                fresh_candidates.append(candidate)
        return fresh_candidates
