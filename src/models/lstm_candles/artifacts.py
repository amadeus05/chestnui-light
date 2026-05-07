from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

import train
from src.models.artifacts import ArtifactStore
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec
from src.models.lstm_candles.walk_forward import LstmCandlesWalkForwardResult


@dataclass(slots=True)
class LstmCandlesArtifactWriter:
    spec: ModelSpec
    artifact_store: ArtifactStore

    def save_walk_forward_result(self, result: LstmCandlesWalkForwardResult) -> ArtifactPaths:
        paths = self.artifact_store.paths_for(self.spec)
        self.artifact_store.ensure_root(paths)

        result.predictions.to_csv(paths.predictions, index=False)
        if result.model_state is not None:
            torch.save(result.model_state, paths.model)
        self.artifact_store.write_metadata(paths, self.build_walk_forward_metadata(result))
        self.artifact_store.write_json(paths.metrics, result.metrics)
        self.artifact_store.write_json(paths.summary, self.build_walk_forward_summary(result))

        return paths

    def build_walk_forward_metadata(self, result: LstmCandlesWalkForwardResult) -> ModelMetadata:
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
                "note": "Candle LSTM walk-forward uses fold-local sequence standardization.",
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

    def build_walk_forward_summary(self, result: LstmCandlesWalkForwardResult) -> dict[str, Any]:
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
    def build_prediction_period(predictions: pd.DataFrame) -> dict[str, str] | None:
        if predictions.empty or train.TIMESTAMP_COLUMN not in predictions.columns:
            return None
        timestamps = pd.to_datetime(predictions[train.TIMESTAMP_COLUMN])
        return {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
        }
