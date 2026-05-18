from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src_refactor.core.contracts import ModelArtifactStore, ModelInputBuilder, ModelPredictor, ModelTrainer
from src_refactor.core.types import ModelSpec, ModelType


@dataclass(frozen=True, slots=True)
class ModelBundle:
    spec: ModelSpec
    input_builder: ModelInputBuilder
    trainer: ModelTrainer
    artifact_store: ModelArtifactStore

    def load_predictor(self, fold_id: int | None = None) -> ModelPredictor:
        return self.artifact_store.load_predictor(self.spec, fold_id=fold_id)


BundleFactory = Callable[[ModelSpec], ModelBundle]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[ModelType, BundleFactory] = {}

    def register(self, model_type: ModelType, factory: BundleFactory) -> None:
        self._factories[model_type] = factory

    def get(self, spec: ModelSpec) -> ModelBundle:
        try:
            factory = self._factories[spec.model_type]
        except KeyError as exc:
            registered = ", ".join(sorted(self._factories)) or "<empty>"
            raise KeyError(f"Model type '{spec.model_type}' is not registered. Registered: {registered}") from exc
        return factory(spec)
