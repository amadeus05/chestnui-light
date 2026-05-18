from __future__ import annotations

from abc import ABC, abstractmethod

from src_refactor.core.contracts.model_predictor import ModelPredictor
from src_refactor.core.types import ModelArtifact, ModelSpec


class ModelArtifactStore(ABC):
    @abstractmethod
    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        raise NotImplementedError

    @abstractmethod
    def load_predictor(self, spec: ModelSpec, fold_id: int | None = None) -> ModelPredictor:
        raise NotImplementedError
