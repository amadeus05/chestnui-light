from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import pandas as pd

from src_refactor.application.pipeline import PipelineStepResult, StoredPredictionSource, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeConfig, build_backtest_runtime, build_runtime
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.contracts.historical_market_feed import HistoricalMarketFeed
from src_refactor.domain.trading import TradingEngineStepResult


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    steps: tuple[PipelineStepResult, ...] = ()
    final_result: TradingEngineStepResult | None = None


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
        steps = self.pipeline.market_cache.update_from_frame(
            frame,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        )
        results: list[PipelineStepResult] = []
        for timestamp, group in groupby(steps, key=lambda item: item.snapshot.current_timestamp):
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            results.append(self.pipeline.on_contexts(list(group)))

        final_result = None
        if close_open_positions:
            final_mark_timestamp, final_mark_prices = self._final_mark_prices(frame, start=start, end=end)
            if final_mark_timestamp is not None and final_mark_prices:
                final_result = self.pipeline.trading_engine.close_all_positions(
                    timestamp=final_mark_timestamp,
                    mark_prices=final_mark_prices,
                    reason="FINAL",
                )

        return BacktestRunResult(steps=tuple(results), final_result=final_result)

    def _final_mark_prices(
        self,
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp | None, dict[str, float]]:
        if frame.empty:
            return None, {}
        filtered = frame.copy()
        filtered[self.timestamp_column] = pd.to_datetime(filtered[self.timestamp_column])
        if start is not None:
            filtered = filtered.loc[filtered[self.timestamp_column] >= pd.to_datetime(start)]
        if end is not None:
            filtered = filtered.loc[filtered[self.timestamp_column] <= pd.to_datetime(end)]
        if filtered.empty:
            return None, {}
        final_timestamp = filtered[self.timestamp_column].max()
        latest_rows = (
            filtered.loc[filtered[self.timestamp_column] == final_timestamp]
            .dropna(subset=["close"])
            .drop_duplicates(subset=[self.symbol_column], keep="last")
        )
        return pd.to_datetime(final_timestamp), {
            str(row[self.symbol_column]): float(row["close"])
            for _, row in latest_rows.iterrows()
        }


def build_oos_backtest_runner(
    *,
    config: RuntimeConfig,
    prediction_store: PredictionStore,
    model_id: str | None = None,
    symbols: tuple[str, ...] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    data_source: HistoricalMarketFeed | None = None,
    broker: BrokerGateway | None = None,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
) -> BacktestRunner:
    predictions = prediction_store.read(
        model_id=model_id or config.model.model_id,
        symbols=symbols or config.symbols or None,
        start=start,
        end=end,
    )
    runtime_adapters = build_backtest_runtime(
        config,
        prediction_source=StoredPredictionSource.from_predictions(predictions),
        data_source=data_source,
        broker=broker,
    )
    runtime = build_runtime(config, runtime_adapters)
    return BacktestRunner(
        pipeline=runtime.pipeline,
        timestamp_column=timestamp_column,
        symbol_column=symbol_column,
    )
