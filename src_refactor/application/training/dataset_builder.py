from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src_refactor.domain.features import FeaturePipeline, FeaturePipelineConfig, FeaturePipelineResult
from src_refactor.domain.labels import LabelingService


@dataclass(frozen=True, slots=True)
class TrainingDatasetBuildResult:
    dataset: pd.DataFrame
    feature_result: FeaturePipelineResult
    feature_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingDatasetBuilder:
    feature_pipeline: FeaturePipeline = field(default_factory=FeaturePipeline)
    labeling_service: LabelingService | None = None

    @classmethod
    def from_configs(
        cls,
        *,
        feature_config: FeaturePipelineConfig | None = None,
        labeling_config: Any,
    ) -> "TrainingDatasetBuilder":
        return cls(
            feature_pipeline=FeaturePipeline(feature_config),
            labeling_service=LabelingService.from_config(labeling_config),
        )

    def build(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame] | None = None,
    ) -> TrainingDatasetBuildResult:
        if self.labeling_service is None:
            raise ValueError("TrainingDatasetBuilder requires labeling_service.")

        feature_result = self.feature_pipeline.build(base_candle_map, htf_candle_map)
        labeled_frames: list[pd.DataFrame] = []
        for symbol, feature_frame in feature_result.feature_map.items():
            labeled = self.labeling_service.build_labeled_frame(feature_frame, list(feature_result.feature_columns))
            labeled["symbol"] = symbol
            labeled_frames.append(labeled)

        dataset = pd.concat(labeled_frames, ignore_index=True) if labeled_frames else pd.DataFrame()
        if not dataset.empty and "timestamp" in dataset.columns:
            dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        elif not dataset.empty and "timestamp_ms" in dataset.columns:
            dataset = dataset.sort_values(["timestamp_ms", "symbol"]).reset_index(drop=True)

        return TrainingDatasetBuildResult(
            dataset=dataset,
            feature_result=feature_result,
            feature_columns=feature_result.feature_columns,
        )
