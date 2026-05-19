from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.application.backtest.metrics import BacktestMetrics, BacktestMetricsCalculator
from src_refactor.application.pipeline import PipelineStepResult, TradingPipeline
from src_refactor.application.runtime_loop import RuntimeLoop, RuntimeLoopResult
from src_refactor.core.types import MarketDataBatch
from src_refactor.domain.trading import TradingEngineStepResult
from src_refactor.infrastructure.feeds import HistoricalFrameMarketStream


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    steps: tuple[PipelineStepResult, ...] = ()
    final_result: TradingEngineStepResult | None = None
    metrics: BacktestMetrics | None = None


@dataclass(frozen=True, slots=True)
class BacktestRunner:
    pipeline: TradingPipeline
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    def run(
        self,
        *,
        frame: pd.DataFrame,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        close_open_positions: bool = True,
    ) -> BacktestRunResult:
        result = self._run_backtest_loop(frame, start=start, end=end)
        final_result = (
            self._close_open_positions(frame, start=start, end=end)
            if close_open_positions
            else None
        )
        portfolio = self.pipeline.trading_engine.portfolio
        metrics = BacktestMetricsCalculator.from_portfolio_state(
            portfolio.state,
            initial_balance=portfolio.initial_balance,
        )
        return BacktestRunResult(steps=result.steps, final_result=final_result, metrics=metrics)

    def _run_backtest_loop(
        self,
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> RuntimeLoopResult:
        if start is None and end is None:
            return RuntimeLoop(
                stream=HistoricalFrameMarketStream(
                    frame=frame,
                    timestamp_column=self.timestamp_column,
                    symbol_column=self.symbol_column,
                ),
                pipeline=self.pipeline,
            ).run()

        steps: list[PipelineStepResult] = []
        for batch in HistoricalFrameMarketStream(
            frame=frame,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        ).stream():
            contexts = self.pipeline.contexts_from_batch(batch)
            contexts = tuple(
                context
                for context in contexts
                if self._timestamp_inside_window(
                    context.snapshot.current_timestamp,
                    start=start,
                    end=end,
                )
            )
            if contexts:
                steps.append(self.pipeline.process_contexts(contexts))
        return RuntimeLoopResult(steps=tuple(steps))

    def _close_open_positions(
        self,
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> TradingEngineStepResult | None:
        if frame.empty:
            return None
        final_batch = self._final_mark_batch(frame, start=start, end=end)
        if final_batch is None:
            return None
        return self.pipeline.trading_engine.close_all_positions(
            timestamp=final_batch.timestamp,
            mark_prices=final_batch.mark_prices,
            reason="FINAL",
        )

    def _final_mark_batch(
        self,
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> MarketDataBatch | None:
        batches = HistoricalFrameMarketStream(
            frame=frame,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        ).stream()
        final_batch = None
        for batch in batches:
            if self._timestamp_inside_window(batch.timestamp, start=start, end=end):
                final_batch = batch
        return final_batch

    @staticmethod
    def _timestamp_inside_window(
        timestamp: pd.Timestamp,
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> bool:
        value = pd.to_datetime(timestamp)
        if start is not None and value < pd.to_datetime(start):
            return False
        if end is not None and value > pd.to_datetime(end):
            return False
        return True
