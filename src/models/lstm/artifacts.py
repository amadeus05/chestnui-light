from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import pandas as pd

import train
from src.models.artifacts import ArtifactStore
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec
from src.models.lstm.trainer import TrainedLstmModel


@dataclass(slots=True)
class LstmArtifactWriter:
    spec: ModelSpec
    artifact_store: ArtifactStore

    def save_production_model(self, trained: TrainedLstmModel) -> ArtifactPaths:
        paths = self.artifact_store.paths_for(self.spec)
        self.artifact_store.ensure_root(paths)

        torch.save(self.build_model_payload(trained), paths.model)
        self.artifact_store.write_metadata(paths, self.build_metadata(trained))
        self.artifact_store.write_json(paths.metrics, self.build_metrics_payload(trained))

        return paths

    def build_model_payload(self, trained: TrainedLstmModel) -> dict[str, Any]:
        return {
            "model_state_dict": trained.model.state_dict(),
            "feature_columns": list(trained.feature_columns),
            "standardizer": trained.standardizer.to_payload(),
            "sequence_length": int(trained.dataset.sequence_length),
            "symbols": self.resolve_symbols(trained),
            "model_args": trained.model_args,
            "training": trained.training_summary,
        }

    def build_metadata(self, trained: TrainedLstmModel) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.spec.key,
            artifact_name=self.spec.artifact_name,
            model_type=self.spec.model_type,
            feature_source=self.spec.feature_source,
            feature_columns=list(trained.feature_columns),
            symbols=self.resolve_symbols(trained),
            label_mapping={"short": 0, "long": 1},
            inverse_label_mapping={str(key): value for key, value in train.CLASS_TO_LABEL.items()},
            event_filter={},
            feature_clip={
                "enabled": False,
                "bounds": {},
                "note": "LSTM uses sequence standardization instead of LightGBM feature clipping.",
            },
            train_period=self.build_train_period(trained),
            sequence_length=int(trained.dataset.sequence_length),
            model_args=trained.model_args,
            standardizer=trained.standardizer.to_payload(),
        )

    def build_metrics_payload(self, trained: TrainedLstmModel) -> dict[str, Any]:
        return {
            "model_key": self.spec.key,
            "artifact_name": self.spec.artifact_name,
            "training_summary": trained.training_summary,
            "feature_count": int(len(trained.feature_columns)),
            "rows": int(len(trained.dataset)),
        }

    @staticmethod
    def resolve_symbols(trained: TrainedLstmModel) -> list[str]:
        frame = trained.dataset.sample_frame
        if train.SYMBOL_COLUMN not in frame.columns:
            return []
        return sorted(frame[train.SYMBOL_COLUMN].astype(str).dropna().unique().tolist())

    @staticmethod
    def build_train_period(trained: TrainedLstmModel) -> dict[str, str] | None:
        frame = trained.dataset.sample_frame
        if frame.empty or train.TIMESTAMP_COLUMN not in frame.columns:
            return None
        timestamps = pd.to_datetime(frame[train.TIMESTAMP_COLUMN])
        return {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
        }
