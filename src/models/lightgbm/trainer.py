from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import train
from src.models.lightgbm.dataset import LightGbmDatasetBuilder, LightGbmDatasetRequest, LightGbmFitFrame


@dataclass(frozen=True, slots=True)
class LightGbmTrainRequest:
    db_path: str
    symbols: list[str]
    seed: int
    n_estimators: int


@dataclass(slots=True)
class TrainedLightGbmModel:
    model: Any
    fit_frame: LightGbmFitFrame
    dataset_diagnostics: dict[str, Any]
    event_filter: dict[str, Any]
    training_summary: dict[str, Any]


class LightGbmTrainer:
    """Production trainer for the LightGBM directional model."""

    def __init__(self, dataset_builder: LightGbmDatasetBuilder | None = None):
        self.dataset_builder = dataset_builder or LightGbmDatasetBuilder()

    def train_production(self, request: LightGbmTrainRequest) -> TrainedLightGbmModel:
        dataset = self.dataset_builder.load_training_dataset(
            LightGbmDatasetRequest(
                db_path=request.db_path,
                symbols=request.symbols,
            )
        )
        fit_frame = self.dataset_builder.build_fit_frame(dataset.frame, dataset.feature_columns)
        model = self.fit_model(fit_frame, seed=request.seed, n_estimators=request.n_estimators)

        return TrainedLightGbmModel(
            model=model,
            fit_frame=fit_frame,
            dataset_diagnostics=dataset.diagnostics,
            event_filter=dataset.event_filter,
            training_summary={
                "rows": int(len(fit_frame.frame)),
                "feature_count": int(len(fit_frame.feature_columns)),
                "n_estimators": int(request.n_estimators),
                "seed": int(request.seed),
            },
        )

    def fit_model(self, fit_frame: LightGbmFitFrame, seed: int, n_estimators: int):
        model = train.build_model(seed=seed, n_estimators=n_estimators)
        sample_weights = self.dataset_builder.build_sample_weights(fit_frame.frame)
        model.fit(
            fit_frame.x,
            fit_frame.y,
            sample_weight=sample_weights,
            categorical_feature=self.resolve_categorical_feature(fit_frame.feature_columns),
        )
        return model

    @staticmethod
    def resolve_categorical_feature(feature_columns: list[str]):
        return [train.SYMBOL_COLUMN] if train.SYMBOL_COLUMN in feature_columns else "auto"
