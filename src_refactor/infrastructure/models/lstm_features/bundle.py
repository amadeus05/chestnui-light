from __future__ import annotations

from src_refactor.core.types import ModelSpec
from src_refactor.infrastructure.models.lstm_features.input_builder import LstmFeatureInputBuilder
from src_refactor.infrastructure.models.lstm_features.trainer import (
    LstmFeatureArtifactStore,
    LstmFeatureTrainer,
)
from src_refactor.infrastructure.models.registry import ModelBundle


def create_lstm_features_bundle(spec: ModelSpec) -> ModelBundle:
    return ModelBundle(
        spec=spec,
        input_builder=LstmFeatureInputBuilder(),
        trainer=LstmFeatureTrainer(),
        artifact_store=LstmFeatureArtifactStore(),
    )
