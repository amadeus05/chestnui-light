from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import ModelInput, ModelSpec


@dataclass(frozen=True, slots=True)
class ModelInputRequest:
    symbol: str
    timeframe: str
    base_candles: pd.DataFrame
    spec: ModelSpec
    htf_candles: pd.DataFrame | None = None
    base_candle_map: dict[str, pd.DataFrame] = field(default_factory=dict)
    htf_candle_map: dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_pipeline: Any | None = None


class ModelInputBuilder(ABC):
    @abstractmethod
    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> ModelInput:
        raise NotImplementedError

    @abstractmethod
    def build_predict_input(self, request: ModelInputRequest | pd.DataFrame, spec: ModelSpec | None = None) -> ModelInput:
        raise NotImplementedError
