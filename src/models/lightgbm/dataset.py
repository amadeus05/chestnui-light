from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config as cfg
import train


@dataclass(frozen=True, slots=True)
class LightGbmDatasetRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))


@dataclass(slots=True)
class LightGbmDatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    event_filter: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class LightGbmFitFrame:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    clip_bounds: dict[str, dict[str, float]]

    @property
    def x(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column]


class LightGbmDatasetBuilder:
    """Dataset/feature preparation for the LightGBM directional model."""

    def load_training_dataset(self, request: LightGbmDatasetRequest) -> LightGbmDatasetBundle:
        frame = train.load_training_frame(request.db_path, request.symbols)
        feature_columns = self.select_feature_columns(frame)
        return LightGbmDatasetBundle(
            frame=frame,
            feature_columns=feature_columns,
            event_filter=dict(frame.attrs.get("event_filter_config") or {}),
            diagnostics=self.build_diagnostics(frame),
        )

    def select_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        return list(train.select_feature_columns(frame))

    def build_fit_frame(self, frame: pd.DataFrame, feature_columns: list[str]) -> LightGbmFitFrame:
        clip_bounds = self.build_clip_bounds(frame, feature_columns)
        clipped = self.apply_clip_bounds(frame, clip_bounds)
        return LightGbmFitFrame(
            frame=clipped,
            feature_columns=list(feature_columns),
            target_column=train.TARGET_COLUMN,
            clip_bounds=clip_bounds,
        )

    def build_clip_bounds(self, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
        return train.build_feature_clip_bounds(frame, feature_columns)

    def apply_clip_bounds(self, frame: pd.DataFrame, clip_bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
        return train.apply_feature_clip_bounds(frame, clip_bounds)

    def build_sample_weights(self, frame: pd.DataFrame):
        return train.compute_sample_weights(frame[train.TIMESTAMP_COLUMN])

    def build_diagnostics(self, frame: pd.DataFrame) -> dict[str, Any]:
        all_timestamps = frame.attrs.get("all_timestamps", [])
        return {
            "rows": int(len(frame)),
            "candidate_rows": int(frame.attrs.get("candidate_rows", len(frame))),
            "excluded_by_event_filter_rows": int(frame.attrs.get("excluded_by_event_filter_rows", 0)),
            "excluded_non_directional_rows": int(frame.attrs.get("excluded_non_directional_rows", 0)),
            "all_timestamps_count": int(len(all_timestamps)),
            "rows_by_symbol": frame.attrs.get("directional_rows_by_symbol", {}),
        }
