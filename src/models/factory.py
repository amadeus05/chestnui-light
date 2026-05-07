from __future__ import annotations

from src.models.artifacts import ArtifactStore
from src.models.base import BaseModelRunner
from src.models.contracts import ModelSpec
from src.models.lightgbm.runner import LightGbmRunner
from src.models.lstm.runner import LstmRunner
from src.models.lstm_candles.runner import LstmCandlesRunner


def create_runner(spec: ModelSpec, artifact_store: ArtifactStore | None = None) -> BaseModelRunner:
    store = artifact_store or ArtifactStore()
    if spec.key == "lightgbm":
        return LightGbmRunner(spec=spec, artifact_store=store)
    if spec.key == "lstm":
        return LstmRunner(spec=spec, artifact_store=store)
    if spec.key == "lstm_candles":
        return LstmCandlesRunner(spec=spec, artifact_store=store)
    raise ValueError(f"No runner registered for model '{spec.key}'.")
