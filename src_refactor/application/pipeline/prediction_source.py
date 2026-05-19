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
from src_refactor.core.contracts.prediction_store import PredictionStore
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

    @classmethod
    def from_store(
        cls,
        store: PredictionStore,
        *,
        model_id: str,
        symbols: tuple[str, ...] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> "StoredPredictionSource":
        return cls.from_predictions(
            store.read(
                model_id=model_id,
                symbols=symbols,
                start=start,
                end=end,
            )
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        model_id: str,
        timeframe: str,
        barrier_frame: pd.DataFrame | None = None,
        timestamp_column: str = "timestamp",
        symbol_column: str = "symbol",
    ) -> "StoredPredictionSource":
        return cls.from_predictions(
            predictions_from_frame(
                frame,
                model_id=model_id,
                timeframe=timeframe,
                barrier_frame=barrier_frame,
                timestamp_column=timestamp_column,
                symbol_column=symbol_column,
            )
        )

    def predictions_for(self, context: MarketContext) -> list[Prediction]:
        return self.predictions_by_timestamp.get(canonical_prediction_timestamp(context.snapshot.current_timestamp), [])


def group_predictions_by_timestamp(predictions: list[Prediction]) -> dict[pd.Timestamp, list[Prediction]]:
    grouped: dict[pd.Timestamp, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[canonical_prediction_timestamp(prediction.timestamp)].append(prediction)
    return dict(grouped)


def predictions_from_frame(
    frame: pd.DataFrame,
    *,
    model_id: str,
    timeframe: str,
    barrier_frame: pd.DataFrame | None = None,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
) -> list[Prediction]:
    prepared = frame.copy()
    _require_columns(prepared, {timestamp_column, symbol_column}, "Prediction frame")
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column])

    if barrier_frame is not None and not {"barrier_stop_pct", "barrier_take_pct"}.issubset(prepared.columns):
        barriers = barrier_frame.copy()
        _require_columns(
            barriers,
            {timestamp_column, symbol_column, "barrier_stop_pct", "barrier_take_pct"},
            "Barrier frame",
        )
        barriers[timestamp_column] = pd.to_datetime(barriers[timestamp_column])
        prepared = prepared.merge(
            barriers[[timestamp_column, symbol_column, "barrier_stop_pct", "barrier_take_pct"]],
            on=[timestamp_column, symbol_column],
            how="left",
        )

    p_long_column = "p_long" if "p_long" in prepared.columns else "proba_long"
    p_short_column = "p_short" if "p_short" in prepared.columns else "proba_short"
    _require_columns(
        prepared,
        {p_long_column, p_short_column, "barrier_stop_pct", "barrier_take_pct"},
        "Prediction frame",
    )
    prepared = prepared.drop_duplicates(subset=[timestamp_column, symbol_column], keep="last")
    prepared = prepared.sort_values([timestamp_column, symbol_column]).reset_index(drop=True)

    predictions: list[Prediction] = []
    for row in prepared.to_dict("records"):
        p_long = float(row[p_long_column])
        p_short = float(row[p_short_column])
        predictions.append(
            Prediction(
                timestamp=canonical_prediction_timestamp(row[timestamp_column]),
                symbol=str(row[symbol_column]),
                timeframe=str(row.get("timeframe", timeframe)),
                model_id=str(row.get("model_id", model_id)),
                direction=int(row.get("direction", 0)),
                confidence=float(row.get("confidence", max(p_long, p_short))),
                fold_id=_optional_int(row.get("fold_id")),
                proba_long=p_long,
                proba_short=p_short,
                signal_gap=abs(p_long - p_short),
                stop_pct=float(row["barrier_stop_pct"]),
                take_pct=float(row["barrier_take_pct"]),
            )
        )
    return predictions


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
        signal_gap=prediction.signal_gap,
        stop_pct=prediction.stop_pct,
        take_pct=prediction.take_pct,
        raw=prediction.raw,
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
