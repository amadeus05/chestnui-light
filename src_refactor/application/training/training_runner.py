from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from src_refactor.application.training.oos_prediction_service import OosPredictionService
from src_refactor.application.training.walk_forward_splitter import WalkForwardSplitter
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import PredictionStore
from src_refactor.core.types import ModelArtifact, Prediction, WalkForwardFold
from src_refactor.infrastructure.models import ModelRegistry


@dataclass(frozen=True, slots=True)
class FoldTrainingResult:
    fold: WalkForwardFold
    artifact: ModelArtifact
    predictions: list[Prediction]


@dataclass(frozen=True, slots=True)
class WalkForwardTrainingResult:
    config: ExperimentConfig
    folds: list[FoldTrainingResult]

    @property
    def predictions(self) -> list[Prediction]:
        return [prediction for fold in self.folds for prediction in fold.predictions]


@dataclass(frozen=True, slots=True)
class WalkForwardTrainingRunner:
    registry: ModelRegistry
    prediction_store: PredictionStore | None = None

    def run(self, frame: pd.DataFrame, config: ExperimentConfig) -> WalkForwardTrainingResult:
        bundle = self.registry.get(config.model)
        splitter = WalkForwardSplitter(config)
        folds = splitter.split(frame)
        timestamps = pd.to_datetime(frame[config.timestamp_column], errors="coerce")

        results: list[FoldTrainingResult] = []
        for fold in folds:
            train_frame = frame.loc[fold.train_mask(timestamps)].copy()
            test_frame = frame.loc[fold.test_mask(timestamps)].copy()
            train_input = bundle.input_builder.build_train_input(train_frame, config)
            artifact = bundle.trainer.train(train_input, config, fold)
            artifact = bundle.artifact_store.save(artifact)
            fold_spec = replace(
                config.model,
                artifact_uri=str(artifact.uri),
                metadata={**artifact.metadata, **config.model.metadata},
            )
            predictor = bundle.artifact_store.load_predictor(fold_spec, fold_id=fold.fold_id)

            prediction_service = OosPredictionService(
                input_builder=bundle.input_builder,
                predictor=predictor,
                spec=fold_spec,
                timestamp_column=config.timestamp_column,
                symbol_column=config.symbol_column,
            )
            predictions = prediction_service.predict_frame(test_frame, fold)
            if self.prediction_store is not None:
                self.prediction_store.write(predictions)

            results.append(FoldTrainingResult(fold=fold, artifact=artifact, predictions=predictions))

        return WalkForwardTrainingResult(config=config, folds=results)
