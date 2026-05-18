from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts import PredictionStore
from src_refactor.core.types import Prediction


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
        for prediction in self._predictions:
            if prediction.model_id != model_id:
                continue
            if symbol_set and prediction.symbol not in symbol_set:
                continue
            if start is not None and prediction.timestamp < start:
                continue
            if end is not None and prediction.timestamp > end:
                continue
            output.append(prediction)
        return sorted(output, key=lambda prediction: (prediction.timestamp, prediction.symbol))
