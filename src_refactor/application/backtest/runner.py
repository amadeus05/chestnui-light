from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import pandas as pd

from src_refactor.application.pipeline import PipelineStepResult, StoredPredictionSource, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeConfig, build_backtest_runtime, build_runtime
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.contracts.historical_market_feed import HistoricalMarketFeed


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    steps: tuple[PipelineStepResult, ...] = ()


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

        return BacktestRunResult(steps=tuple(results))


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
