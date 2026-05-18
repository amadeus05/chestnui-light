from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts import PredictionStore
from src_refactor.core.types import Prediction
from src_refactor.infrastructure.predictions.timestamps import canonical_prediction_timestamp


@dataclass(slots=True)
class InMemoryPredictionStore(PredictionStore):
    """Dev/test store. Use ParquetPredictionStore for persisted OOS predictions."""

    _predictions: list[Prediction] = field(default_factory=list)

    def write(self, predictions: Iterable[Prediction]) -> None:
        self._predictions.extend(predictions)

    def read(
        self,
        *,
        model_id: str,
        symbols: tuple[str, ...] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[Prediction]:
        output: list[Prediction] = []
        symbol_set = set(symbols or ())
        start_key = canonical_prediction_timestamp(start) if start is not None else None
        end_key = canonical_prediction_timestamp(end) if end is not None else None
        for prediction in self._predictions:
            if prediction.model_id != model_id:
                continue
            if symbol_set and prediction.symbol not in symbol_set:
                continue
            prediction_key = canonical_prediction_timestamp(prediction.timestamp)
            if start_key is not None and prediction_key < start_key:
                continue
            if end_key is not None and prediction_key > end_key:
                continue
            output.append(prediction)
        return sorted(output, key=lambda prediction: (canonical_prediction_timestamp(prediction.timestamp), prediction.symbol))
