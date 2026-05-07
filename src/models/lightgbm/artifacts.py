from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

import train
from src.models.artifacts import ArtifactStore
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec
from src.models.lightgbm.trainer import TrainedLightGbmModel


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
    def build_period_payload(frame: pd.DataFrame) -> dict[str, str] | None:
        if frame.empty:
            return None
        timestamps = pd.to_datetime(frame[train.TIMESTAMP_COLUMN])
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
