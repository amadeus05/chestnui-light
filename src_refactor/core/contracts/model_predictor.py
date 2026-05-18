from __future__ import annotations

from abc import ABC, abstractmethod

from src_refactor.core.types import ModelInput, Prediction


class ModelPredictor(ABC):
    @abstractmethod
    def predict(self, model_input: ModelInput) -> Prediction:
        raise NotImplementedError
