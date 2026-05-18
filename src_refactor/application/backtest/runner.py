from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.application.backtest.metrics import BacktestMetrics, BacktestMetricsCalculator
from src_refactor.application.pipeline import PipelineStepResult, StoredPredictionSource, TradingPipeline
from src_refactor.application.runtime_builder import RuntimeConfig, build_backtest_runtime, build_runtime
from src_refactor.application.runtime_loop import RuntimeLoop
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.contracts.historical_market_feed import HistoricalMarketFeed
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
        result = RuntimeLoop(
            stream=HistoricalFrameMarketStream(
                frame=frame,
                timestamp_column=self.timestamp_column,
                symbol_column=self.symbol_column,
            ),
            pipeline=self.pipeline,
            close_open_positions=close_open_positions,
            start=start,
            end=end,
        ).run()
        portfolio = self.pipeline.trading_engine.portfolio
        metrics = BacktestMetricsCalculator.from_portfolio_state(
            portfolio.state,
            initial_balance=portfolio.initial_balance,
        )
        return BacktestRunResult(steps=result.steps, final_result=result.final_result, metrics=metrics)


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
