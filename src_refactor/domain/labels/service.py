from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src_refactor.domain.labels.barriers import attach_barrier_columns
from src_refactor.domain.labels.config import LabelingConfig, ensure_labeling_config
from src_refactor.domain.labels.horizons import compute_effective_horizons
from src_refactor.domain.labels.triple_barrier import triple_barrier_labeling

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]
TARGET_COLUMN = "Target"


@dataclass(frozen=True, slots=True)
class LabelingService:
    config: LabelingConfig

    @classmethod
    def from_config(cls, config: Any) -> "LabelingService":
        return cls(config=ensure_labeling_config(config))

    def attach_barriers(self, frame: pd.DataFrame) -> pd.DataFrame:
        return attach_barrier_columns(frame, self.config)

    def label(self, frame: pd.DataFrame) -> pd.DataFrame:
        return triple_barrier_labeling(frame, self.config)

    def build_labeled_frame(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        with_barriers = self.attach_barriers(frame)
        with_labels = self.label(with_barriers)
        return self.finalize(with_labels, feature_columns)

    def finalize(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        return finalize_labeled_feature_frame(frame, feature_columns, self.config)


def build_labeled_feature_frame(frame: pd.DataFrame, feature_columns: list[str], config: Any) -> pd.DataFrame:
    return LabelingService.from_config(config).build_labeled_frame(frame, feature_columns)


def finalize_labeled_feature_frame(frame: pd.DataFrame, feature_columns: list[str], config: Any) -> pd.DataFrame:
    label_config = ensure_labeling_config(config)
    output = frame.copy()
    effective_horizons = compute_effective_horizons(output, label_config)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else label_config.base_horizon

    output_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + [TARGET_COLUMN]
    if max_horizon > 0:
        if len(output) <= max_horizon:
            return output.iloc[0:0].reindex(columns=output_columns).copy()
        output = output.iloc[:-max_horizon].copy()

    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    output.dropna(inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def build_labeling_snapshot(config: Any) -> dict[str, object]:
    return ensure_labeling_config(config).snapshot()
