from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from src_refactor.application.pipeline.runtime_market_cache import (
    MarketContext,
    candle_map_to_frame_map,
    candles_to_frame,
)
from src_refactor.core.contracts.model_input_builder import ModelInputBuilder, ModelInputRequest
from src_refactor.core.contracts.model_predictor import ModelPredictor
from src_refactor.core.types import ModelSpec, Prediction
from src_refactor.domain.features import FeaturePipeline
from src_refactor.infrastructure.predictions.timestamps import canonical_prediction_timestamp


class PredictionSource:
    def predictions_for(self, context: MarketContext) -> list[Prediction]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ModelPredictionSource(PredictionSource):
    input_builder: ModelInputBuilder
    predictor: ModelPredictor
    model_spec: ModelSpec
    feature_pipeline: FeaturePipeline | None = None
    htf_candle_map: dict[str, pd.DataFrame] = field(default_factory=dict)

    def predictions_for(self, context: MarketContext) -> list[Prediction]:
        base_candle_map = candle_map_to_frame_map(context.base_history_map)
        if context.symbol not in base_candle_map:
            base_candle_map[context.symbol] = candles_to_frame(context.history)

        htf_candle_map = {
            **self.htf_candle_map,
            **candle_map_to_frame_map(context.htf_history_map),
        }
        model_input = self.input_builder.build_predict_input(
            ModelInputRequest(
                symbol=context.symbol,
                timeframe=self.model_spec.timeframe,
                base_candles=base_candle_map[context.symbol],
                spec=self.model_spec,
                htf_candles=htf_candle_map.get(context.symbol),
                base_candle_map=base_candle_map,
                htf_candle_map=htf_candle_map,
                feature_pipeline=self.feature_pipeline,
            )
        )
        return [
            normalize_prediction(
                self.predictor.predict(model_input),
                context=context,
                model_id=self.model_spec.model_id,
            )
        ]


@dataclass(frozen=True, slots=True)
class StoredPredictionSource(PredictionSource):
    predictions_by_timestamp: dict[pd.Timestamp, list[Prediction]] = field(default_factory=dict)

    @classmethod
    def from_predictions(cls, predictions: list[Prediction]) -> "StoredPredictionSource":
        return cls(predictions_by_timestamp=group_predictions_by_timestamp(predictions))

    def predictions_for(self, context: MarketContext) -> list[Prediction]:
        return self.predictions_by_timestamp.get(canonical_prediction_timestamp(context.snapshot.current_timestamp), [])


def group_predictions_by_timestamp(predictions: list[Prediction]) -> dict[pd.Timestamp, list[Prediction]]:
    grouped: dict[pd.Timestamp, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[canonical_prediction_timestamp(prediction.timestamp)].append(prediction)
    return dict(grouped)


def normalize_prediction(prediction: Prediction, *, context: MarketContext, model_id: str) -> Prediction:
    return Prediction(
        timestamp=canonical_prediction_timestamp(context.snapshot.current_timestamp),
        symbol=context.symbol,
        timeframe=context.history[-1].timeframe,
        model_id=model_id,
        direction=prediction.direction,
        confidence=prediction.confidence,
        fold_id=prediction.fold_id,
        proba_long=prediction.proba_long,
        proba_short=prediction.proba_short,
        raw=prediction.raw,
    )
