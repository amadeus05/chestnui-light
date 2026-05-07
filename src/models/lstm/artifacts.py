from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import pandas as pd

import train
from src.models.artifacts import ArtifactStore
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec
from src.models.lstm.trainer import TrainedLstmModel
from src.models.lstm.walk_forward import LstmWalkForwardResult


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

    def save_walk_forward_result(self, result: LstmWalkForwardResult) -> ArtifactPaths:
        paths = self.artifact_store.paths_for(self.spec)
        self.artifact_store.ensure_root(paths)

        result.predictions.to_csv(paths.predictions, index=False)
        if result.model_state is not None:
            torch.save(result.model_state, paths.model)
        self.artifact_store.write_metadata(paths, self.build_walk_forward_metadata(result))
        self.artifact_store.write_json(paths.metrics, result.metrics)
        self.artifact_store.write_json(paths.summary, self.build_walk_forward_summary(result))

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

    def build_walk_forward_metadata(self, result: LstmWalkForwardResult) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.spec.key,
            artifact_name=self.spec.artifact_name,
            model_type=self.spec.model_type,
            feature_source=self.spec.feature_source,
            feature_columns=list(result.feature_columns),
            symbols=list(result.symbols),
            label_mapping={"short": 0, "long": 1},
            inverse_label_mapping={str(key): value for key, value in train.CLASS_TO_LABEL.items()},
            event_filter={},
            feature_clip={
                "enabled": False,
                "bounds": {},
                "note": "LSTM walk-forward uses fold-local sequence standardization.",
            },
            train_period=self.build_prediction_period(result.predictions),
            walk_forward={
                "n_splits": int(result.request.n_splits),
                "purge_gap": int(result.request.purge_gap),
                "split_mode": str(result.request.split_mode),
                "monthly_train_months": int(result.request.monthly_train_months),
                "monthly_test_months": int(result.request.monthly_test_months),
                "monthly_window_mode": str(result.request.monthly_window_mode),
                "fold_count": int(len(result.fold_details)),
                "prediction_rows": int(len(result.predictions)),
            },
            sequence_length=int(result.request.sequence_length),
            model_args={
                "hidden_size": int(result.request.hidden_size),
                "num_layers": int(result.request.num_layers),
                "dropout": float(result.request.dropout),
                "input_size": int(len(result.feature_columns)),
            },
        )

    def build_walk_forward_summary(self, result: LstmWalkForwardResult) -> dict[str, Any]:
        return {
            "model_key": self.spec.key,
            "artifact_name": self.spec.artifact_name,
            "symbols": list(result.symbols),
            "prediction_rows": int(len(result.predictions)),
            "prediction_period": self.build_prediction_period(result.predictions),
            "feature_count": int(len(result.feature_columns)),
            "request": {
                "seed": int(result.request.seed),
                "n_splits": int(result.request.n_splits),
                "split_mode": str(result.request.split_mode),
                "monthly_train_months": int(result.request.monthly_train_months),
                "monthly_test_months": int(result.request.monthly_test_months),
                "monthly_window_mode": str(result.request.monthly_window_mode),
                "purge_gap": int(result.request.purge_gap),
                "sequence_length": int(result.request.sequence_length),
                "batch_size": int(result.request.batch_size),
                "epochs": int(result.request.epochs),
            },
            "fold_details": result.fold_details,
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

    @staticmethod
    def build_prediction_period(predictions) -> dict[str, str] | None:
        if predictions.empty or train.TIMESTAMP_COLUMN not in predictions.columns:
            return None
        timestamps = pd.to_datetime(predictions[train.TIMESTAMP_COLUMN])
        return {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
        }
