from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.core.contracts import ModelInputBuilder, ModelPredictor
from src_refactor.core.types import ModelSpec, Prediction, WalkForwardFold


@dataclass(frozen=True, slots=True)
class OosPredictionService:
    input_builder: ModelInputBuilder
    predictor: ModelPredictor
    spec: ModelSpec
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"

    def predict_frame(self, frame: pd.DataFrame, fold: WalkForwardFold) -> list[Prediction]:
        if frame.empty:
            return []

        predictions: list[Prediction] = []
        for _, row in frame.sort_values(self.timestamp_column).iterrows():
            row_frame = row.to_frame().T
            model_input = self.input_builder.build_predict_input(row_frame, self.spec)
            prediction = self.predictor.predict(model_input)
            predictions.append(
                Prediction(
                    timestamp=pd.to_datetime(row[self.timestamp_column]),
                    symbol=str(row.get(self.symbol_column, prediction.symbol)),
                    timeframe=self.spec.timeframe,
                    model_id=self.spec.model_id,
                    direction=prediction.direction,
                    confidence=prediction.confidence,
                    fold_id=fold.fold_id,
                    proba_long=prediction.proba_long,
                    proba_short=prediction.proba_short,
                    signal_gap=prediction.signal_gap,
                    stop_pct=_optional_float(row.get("barrier_stop_pct")),
                    take_pct=_optional_float(row.get("barrier_take_pct")),
                    raw=prediction.raw,
                )
            )
        return predictions


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
