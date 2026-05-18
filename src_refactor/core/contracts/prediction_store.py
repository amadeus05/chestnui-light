from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

from src_refactor.core.types import Prediction


class PredictionStore(ABC):
    @abstractmethod
    def write(self, predictions: Iterable[Prediction]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        *,
        model_id: str,
        symbols: tuple[str, ...] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[Prediction]:
        raise NotImplementedError
