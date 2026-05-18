from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import ModelInput, ModelSpec


class ModelInputBuilder(ABC):
    @abstractmethod
    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> ModelInput:
        raise NotImplementedError

    @abstractmethod
    def build_predict_input(self, frame: pd.DataFrame, spec: ModelSpec) -> ModelInput:
        raise NotImplementedError
