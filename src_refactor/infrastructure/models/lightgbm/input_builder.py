from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelInputBuilder
from src_refactor.core.types import ModelSpec
from src_refactor.core.types import LightGbmInput
from src_refactor.infrastructure.models.lightgbm.config import LightGbmTrainingConfig

RESERVED_COLUMNS = {"Target", "timestamp", "barrier_stop_pct", "barrier_take_pct"}
EXCLUDED_RAW_FEATURE_COLUMNS = {"open", "high", "low", "close", "volume"}
SYMBOL_COLUMN = "symbol"


@dataclass(frozen=True, slots=True)
class LightGbmInputBuilder(ModelInputBuilder):
    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> LightGbmInput:
        lightgbm_config = LightGbmTrainingConfig.from_metadata(config.model.metadata)
        feature_names = self.select_feature_columns(frame, lightgbm_config)
        clip_bounds = self.build_feature_clip_bounds(frame, feature_names, lightgbm_config)
        prepared = self.apply_feature_clip_bounds(frame, clip_bounds)
        return LightGbmInput(
            features=prepared.loc[:, feature_names].copy(),
            feature_names=tuple(feature_names),
            metadata={
                "target": prepared[config.target_column].copy() if config.target_column in prepared.columns else None,
                "frame": prepared,
                "clip_bounds": clip_bounds,
                "config": lightgbm_config,
            },
        )

    def build_predict_input(self, frame: pd.DataFrame, spec: ModelSpec) -> LightGbmInput:
        feature_names = tuple(spec.metadata.get("feature_columns", ()))
        clip_bounds = dict(spec.metadata.get("feature_clip", {}).get("bounds", {}))
        prepared = self.apply_feature_clip_bounds(frame, clip_bounds)
        if not feature_names:
            lightgbm_config = LightGbmTrainingConfig.from_metadata(spec.metadata)
            feature_names = tuple(self.select_feature_columns(prepared, lightgbm_config))
        missing = [column for column in feature_names if column not in prepared.columns]
        if missing:
            raise ValueError("LightGBM input is missing feature columns: " + ", ".join(missing))
        return LightGbmInput(
            features=prepared.loc[:, list(feature_names)].copy(),
            feature_names=tuple(feature_names),
            metadata={"frame": prepared},
        )

    @staticmethod
    def select_feature_columns(dataset: pd.DataFrame, config: LightGbmTrainingConfig) -> list[str]:
        feature_columns: list[str] = []
        disabled_feature_columns = set(config.disabled_feature_columns)
        for column in dataset.columns:
            if column in RESERVED_COLUMNS:
                continue
            if column in EXCLUDED_RAW_FEATURE_COLUMNS:
                continue
            if column in disabled_feature_columns:
                continue
            if column == SYMBOL_COLUMN:
                if config.use_symbol_feature:
                    feature_columns.append(column)
                continue
            if pd.api.types.is_numeric_dtype(dataset[column]):
                feature_columns.append(column)
        if not feature_columns:
            raise RuntimeError("No usable feature columns found in the dataset.")
        return feature_columns

    @staticmethod
    def build_feature_clip_bounds(
        train_df: pd.DataFrame,
        feature_columns: list[str] | tuple[str, ...],
        config: LightGbmTrainingConfig,
    ) -> dict[str, dict[str, float]]:
        if not config.enable_feature_clip:
            return {}
        lower_q = config.feature_clip_lower_q
        upper_q = config.feature_clip_upper_q
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("Feature clip quantiles must satisfy 0 <= lower < upper <= 1.")

        clip_bounds: dict[str, dict[str, float]] = {}
        for column in feature_columns:
            if column == SYMBOL_COLUMN or not pd.api.types.is_numeric_dtype(train_df[column]):
                continue
            series = train_df[column].replace([float("inf"), float("-inf")], pd.NA).dropna()
            if series.empty:
                continue
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            if pd.isna(lower) or pd.isna(upper):
                continue
            clip_bounds[column] = {"lower": float(lower), "upper": float(upper)}
        return clip_bounds

    @staticmethod
    def apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
        if not clip_bounds:
            return frame
        clipped = frame.copy()
        for column, bounds in clip_bounds.items():
            if column not in clipped.columns:
                continue
            clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
        return clipped
