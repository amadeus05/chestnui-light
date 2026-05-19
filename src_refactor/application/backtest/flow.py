from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.application.backtest.runner import BacktestRunner, BacktestRunResult
from src_refactor.application.pipeline import StoredPredictionSource
from src_refactor.application.runtime_builder import RuntimeConfig, build_backtest_runtime, build_runtime
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader


@dataclass(frozen=True, slots=True)
class StoredPredictionBacktestRequest:
    symbols: tuple[str, ...] = ()
    timeframe: str | None = None
    model_id: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    close_open_positions: bool = True


@dataclass(frozen=True, slots=True)
class StoredPredictionBacktestFlow:
    config: RuntimeConfig
    candle_loader: HistoricalCandleFrameLoader
    prediction_store: PredictionStore
    broker: BrokerGateway | None = None
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    def run(self, request: StoredPredictionBacktestRequest | None = None) -> BacktestRunResult:
        request = request or StoredPredictionBacktestRequest()
        symbols = request.symbols or self.config.symbols
        if not symbols:
            raise ValueError("StoredPredictionBacktestFlow requires symbols in request or RuntimeConfig.")

        model_id = request.model_id or self.config.model.model_id
        timeframe = request.timeframe or self.config.model.timeframe
        market_frame = self.candle_loader.load_symbols(symbols, timeframe)
        prediction_source = StoredPredictionSource.from_store(
            self.prediction_store,
            model_id=model_id,
            symbols=symbols,
            start=request.start,
            end=request.end,
        )
        runtime_adapters = build_backtest_runtime(
            self.config,
            prediction_source=prediction_source,
            broker=self.broker,
        )
        runtime = build_runtime(self.config, runtime_adapters)
        return BacktestRunner(
            pipeline=runtime.pipeline,
            timestamp_column=self.timestamp_column,
            symbol_column=self.symbol_column,
        ).run(
            frame=market_frame,
            start=request.start,
            end=request.end,
            close_open_positions=request.close_open_positions,
        )
