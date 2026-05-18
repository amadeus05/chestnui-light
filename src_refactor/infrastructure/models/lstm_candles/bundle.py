from __future__ import annotations

from src_refactor.core.types import ModelSpec
from src_refactor.infrastructure.models.lstm_candles.input_builder import LstmCandleInputBuilder
from src_refactor.infrastructure.models.lstm_candles.trainer import (
    LstmCandleArtifactStore,
    LstmCandleTrainer,
)
from src_refactor.infrastructure.models.registry import ModelBundle


def create_lstm_candles_bundle(spec: ModelSpec) -> ModelBundle:
    return ModelBundle(
        spec=spec,
        input_builder=LstmCandleInputBuilder(),
        trainer=LstmCandleTrainer(),
        artifact_store=LstmCandleArtifactStore(),
    )
