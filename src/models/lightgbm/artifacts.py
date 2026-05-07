from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

import train
from src.models.artifacts import ArtifactStore
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec
from src.models.lightgbm.trainer import TrainedLightGbmModel
from src.models.lightgbm.walk_forward import LightGbmWalkForwardResult


@dataclass(slots=True)
class LightGbmArtifactWriter:
    spec: ModelSpec
    artifact_store: ArtifactStore

    def save_production_model(self, trained: TrainedLightGbmModel) -> ArtifactPaths:
        paths = self.artifact_store.paths_for(self.spec)
        self.artifact_store.ensure_root(paths)

        joblib.dump(trained.model, paths.model)
        self.artifact_store.write_metadata(paths, self.build_metadata(trained))
        self.artifact_store.write_json(paths.metrics, self.build_metrics_payload(trained))
        self.write_feature_importance(trained, paths)

        return paths

    def save_walk_forward_result(self, result: LightGbmWalkForwardResult) -> ArtifactPaths:
        paths = self.artifact_store.paths_for(self.spec)
        self.artifact_store.ensure_root(paths)

        result.predictions.to_csv(paths.predictions, index=False)
        self.artifact_store.write_json(paths.summary, self.build_walk_forward_summary(result))
        self.artifact_store.write_metadata(paths, self.build_walk_forward_metadata(result))

        return paths

    def build_metadata(self, trained: TrainedLightGbmModel) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.spec.key,
            artifact_name=self.spec.artifact_name,
            model_type=self.spec.model_type,
            feature_source=self.spec.feature_source,
            feature_columns=list(trained.fit_frame.feature_columns),
            symbols=self.resolve_symbols(trained),
            label_mapping={"short": 0, "long": 1},
            inverse_label_mapping={str(key): value for key, value in train.CLASS_TO_LABEL.items()},
            event_filter=trained.event_filter,
            feature_clip={
                "enabled": bool(trained.fit_frame.clip_bounds),
                "bounds": trained.fit_frame.clip_bounds,
            },
            train_period=self.build_period_payload(trained.fit_frame.frame),
            model_args={
                "n_estimators": int(getattr(trained.model, "n_estimators", 0) or 0),
                "best_iteration": int(getattr(trained.model, "best_iteration_", 0) or 0),
            },
        )

    def build_metrics_payload(self, trained: TrainedLightGbmModel) -> dict[str, Any]:
        return {
            "model_key": self.spec.key,
            "artifact_name": self.spec.artifact_name,
            "training_summary": trained.training_summary,
            "dataset_diagnostics": trained.dataset_diagnostics,
            "feature_count": int(len(trained.fit_frame.feature_columns)),
            "rows": int(len(trained.fit_frame.frame)),
        }

    def build_walk_forward_metadata(self, result: LightGbmWalkForwardResult) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.spec.key,
            artifact_name=self.spec.artifact_name,
            model_type=self.spec.model_type,
            feature_source=self.spec.feature_source,
            feature_columns=list(result.feature_columns),
            symbols=list(result.symbols),
            label_mapping={"short": 0, "long": 1},
            inverse_label_mapping={str(key): value for key, value in train.CLASS_TO_LABEL.items()},
            event_filter=result.event_filter,
            feature_clip={
                "enabled": False,
                "bounds": {},
                "note": "Fold-specific clipping is applied inside the walk-forward prediction builder.",
            },
            train_period=self.build_period_payload(result.predictions, timestamp_column="timestamp"),
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
        )

    def build_walk_forward_summary(self, result: LightGbmWalkForwardResult) -> dict[str, Any]:
        prediction_period = self.build_period_payload(result.predictions, timestamp_column="timestamp")
        return {
            "model_key": self.spec.key,
            "artifact_name": self.spec.artifact_name,
            "symbols": list(result.symbols),
            "prediction_rows": int(len(result.predictions)),
            "prediction_period": prediction_period,
            "feature_count": int(len(result.feature_columns)),
            "request": {
                "seed": int(result.request.seed),
                "n_splits": int(result.request.n_splits),
                "split_mode": str(result.request.split_mode),
                "monthly_train_months": int(result.request.monthly_train_months),
                "monthly_test_months": int(result.request.monthly_test_months),
                "monthly_window_mode": str(result.request.monthly_window_mode),
                "purge_gap": int(result.request.purge_gap),
            },
            "fold_details": result.fold_details,
        }

    def write_feature_importance(self, trained: TrainedLightGbmModel, paths: ArtifactPaths) -> None:
        if paths.feature_importance is None or not hasattr(trained.model, "booster_"):
            return
        importance = pd.DataFrame(
            {
                "feature": trained.fit_frame.feature_columns,
                "importance_gain": trained.model.booster_.feature_importance(importance_type="gain"),
                "importance_split": trained.model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        importance.to_csv(paths.feature_importance, index=False)

    @staticmethod
    def build_period_payload(frame: pd.DataFrame, timestamp_column: str = train.TIMESTAMP_COLUMN) -> dict[str, str] | None:
        if frame.empty or timestamp_column not in frame.columns:
            return None
        timestamps = pd.to_datetime(frame[timestamp_column])
        return {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
        }

    @staticmethod
    def resolve_symbols(trained: TrainedLightGbmModel) -> list[str]:
        frame = trained.fit_frame.frame
        if train.SYMBOL_COLUMN not in frame.columns:
            return []
        return sorted(frame[train.SYMBOL_COLUMN].astype(str).dropna().unique().tolist())
