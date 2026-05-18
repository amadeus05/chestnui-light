from __future__ import annotations

from abc import ABC, abstractmethod

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import ModelArtifact, ModelInput, WalkForwardFold


class ModelTrainer(ABC):
    @abstractmethod
    def train(
        self,
        train_input: ModelInput,
        config: ExperimentConfig,
        fold: WalkForwardFold | None = None,
    ) -> ModelArtifact:
        raise NotImplementedError
