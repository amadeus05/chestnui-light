from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src_refactor.application.training.dataset_builder import TrainingDatasetBuilder
from src_refactor.application.training.training_runner import WalkForwardTrainingResult, WalkForwardTrainingRunner
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import ModelSpec
from src_refactor.domain.features import FeaturePipelineConfig
from src_refactor.domain.labels import ensure_labeling_config
from src_refactor.infrastructure.models.default_registry import create_default_model_registry
from src_refactor.infrastructure.predictions import ParquetPredictionStore


@dataclass(frozen=True, slots=True)
class WvfOosRunConfig:
    model_type: str = "lightgbm"
    timeframe: str = "1h"
    profile: str = "baseline"
    symbols: tuple[str, ...] = ()
    split_mode: str = "monthly_expanding"
    n_splits: int = 5
    train_months: int = 6
    test_months: int = 1
    purge_gap: int = 0
    predictions_path: Path = Path("models/predictions/oos_predictions.parquet")
    feature_request: dict | None = None
    feature_profiles: dict[str, object] | None = None
    labeling_config: Any = None
    model_metadata: dict[str, Any] | None = None


def run_wvf_oos(
    *,
    base_candle_map: dict[str, pd.DataFrame],
    htf_candle_map: dict[str, pd.DataFrame] | None = None,
    config: WvfOosRunConfig,
) -> WalkForwardTrainingResult:
    labeling_config = ensure_labeling_config(config.labeling_config or {})
    feature_config = FeaturePipelineConfig(
        raw_request=config.feature_request,
        profile_map=config.feature_profiles,
    )
    dataset_result = TrainingDatasetBuilder.from_configs(
        feature_config=feature_config,
        labeling_config=labeling_config,
    ).build(base_candle_map, htf_candle_map)

    model_spec = ModelSpec(
        model_type=config.model_type,  # type: ignore[arg-type]
        timeframe=config.timeframe,
        profile=config.profile,
        symbols=config.symbols,
        metadata={
            **dict(config.model_metadata or {}),
            "labeling": labeling_config.to_mapping(),
        },
    )
    experiment_config = ExperimentConfig(
        model=model_spec,
        split_mode=config.split_mode,  # type: ignore[arg-type]
        symbols=config.symbols,
        n_splits=config.n_splits,
        train_months=config.train_months,
        test_months=config.test_months,
        purge_gap=config.purge_gap,
    )
    prediction_store = ParquetPredictionStore(config.predictions_path)
    runner = WalkForwardTrainingRunner(
        registry=create_default_model_registry(),
        prediction_store=prediction_store,
    )
    return runner.run(dataset_result.dataset, experiment_config)
