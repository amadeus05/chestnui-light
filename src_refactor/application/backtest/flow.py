from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.application.backtest.runner import BacktestRunner, BacktestRunResult
from src_refactor.application.pipeline import StoredPredictionSource
from src_refactor.application.runtime_builder import RuntimeConfig, build_runtime, build_runtime_adapters
from src_refactor.application.training.training_runner import WalkForwardTrainingResult
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.contracts.broker_gateway import BrokerGateway
from src_refactor.core.types import Prediction
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader
from src_refactor.infrastructure.predictions import ParquetPredictionStore


@dataclass(frozen=True, slots=True)
class StoredPredictionBacktestRequest:
    symbols: tuple[str, ...] = ()
    timeframe: str | None = None
    model_id: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    close_open_positions: bool = True

    @classmethod
    def from_training_result(
        cls,
        result: WalkForwardTrainingResult,
        *,
        close_open_positions: bool = True,
    ) -> "StoredPredictionBacktestRequest":
        return cls.from_predictions(
            result.predictions,
            symbols=result.config.symbols,
            timeframe=result.config.model.timeframe,
            model_id=result.config.model.model_id,
            close_open_positions=close_open_positions,
        )

    @classmethod
    def from_predictions(
        cls,
        predictions: list[Prediction],
        *,
        symbols: tuple[str, ...] = (),
        timeframe: str | None = None,
        model_id: str | None = None,
        close_open_positions: bool = True,
    ) -> "StoredPredictionBacktestRequest":
        if not predictions:
            return cls(
                symbols=symbols,
                timeframe=timeframe,
                model_id=model_id,
                close_open_positions=close_open_positions,
            )
        timestamps = [pd.to_datetime(prediction.timestamp) for prediction in predictions]
        return cls(
            symbols=symbols or tuple(dict.fromkeys(prediction.symbol for prediction in predictions)),
            timeframe=timeframe or predictions[0].timeframe,
            model_id=model_id or predictions[0].model_id,
            start=min(timestamps),
            end=max(timestamps),
            close_open_positions=close_open_positions,
        )


@dataclass(frozen=True, slots=True)
class StoredPredictionBacktestFlow:
    config: RuntimeConfig
    candle_loader: HistoricalCandleFrameLoader
    prediction_store: PredictionStore
    broker: BrokerGateway | None = None
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    @classmethod
    def from_training_result(
        cls,
        *,
        config: RuntimeConfig,
        candle_loader: HistoricalCandleFrameLoader,
        result: WalkForwardTrainingResult,
        broker: BrokerGateway | None = None,
        timestamp_column: str = "timestamp",
        symbol_column: str = "symbol",
    ) -> "StoredPredictionBacktestFlow":
        if result.prediction_store_path is None:
            raise ValueError("WalkForwardTrainingResult does not include prediction_store_path.")
        return cls(
            config=config,
            candle_loader=candle_loader,
            prediction_store=ParquetPredictionStore(result.prediction_store_path),
            broker=broker,
            timestamp_column=timestamp_column,
            symbol_column=symbol_column,
        )

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
        runtime_adapters = build_runtime_adapters(
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
