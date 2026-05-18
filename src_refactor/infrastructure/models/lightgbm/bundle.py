from __future__ import annotations

from src_refactor.core.types import ModelSpec
from src_refactor.infrastructure.models.lightgbm.input_builder import LightGbmInputBuilder
from src_refactor.infrastructure.models.lightgbm.trainer import LightGbmArtifactStore, LightGbmTrainer
from src_refactor.infrastructure.models.registry import ModelBundle


def create_lightgbm_bundle(spec: ModelSpec) -> ModelBundle:
    return ModelBundle(
        spec=spec,
        input_builder=LightGbmInputBuilder(),
        trainer=LightGbmTrainer(),
        artifact_store=LightGbmArtifactStore(),
    )
