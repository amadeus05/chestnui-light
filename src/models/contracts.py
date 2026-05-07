from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import pandas as pd


ModelKey = Literal["lightgbm", "lstm", "lstm_candles"]
ModelType = Literal["lightgbm", "lstm", "lstm_candles"]
FeatureSource = Literal["etl_features", "etl_sequences", "candle_sequences"]
RunMode = Literal["train", "wfv", "backtest", "production"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: ModelKey
    display_name: str
    artifact_name: str
    model_type: ModelType
    feature_source: FeatureSource
    supports_production: bool
    supports_walk_forward: bool
    supports_backtest_replay: bool
    default_predictions_name: str
    default_chart_name: str
    legacy_train_module: str | None = None
    legacy_wfv_module: str | None = None
    legacy_backtest_module: str | None = None
    legacy_production_module: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    root: Path
    model: Path
    metadata: Path
    metrics: Path
    predictions: Path
    summary: Path
    feature_formulas: Path
    feature_importance: Path | None = None
    fold_feature_importance: Path | None = None


@dataclass(slots=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    fold_details: list[dict[str, Any]] = field(default_factory=list)
    model_state: Any | None = None


@dataclass(slots=True)
class ModelMetadata:
    model_key: str
    artifact_name: str
    model_type: str
    feature_source: str
    feature_columns: list[str]
    symbols: list[str]
    label_mapping: dict[str, int]
    inverse_label_mapping: dict[str, int]
    event_filter: dict[str, Any]
    feature_clip: dict[str, Any]
    train_period: dict[str, str] | None = None
    walk_forward: dict[str, Any] | None = None
    sequence_length: int | None = None
    model_args: dict[str, Any] | None = None
    standardizer: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "artifact_name": self.artifact_name,
            "model_type": self.model_type,
            "feature_source": self.feature_source,
            "feature_columns": self.feature_columns,
            "symbols": self.symbols,
            "label_mapping": self.label_mapping,
            "inverse_label_mapping": self.inverse_label_mapping,
            "event_filter": self.event_filter,
            "feature_clip": self.feature_clip,
            "train_period": self.train_period,
            "walk_forward": self.walk_forward,
            "sequence_length": self.sequence_length,
            "model_args": self.model_args,
            "standardizer": self.standardizer,
        }


class ModelRunner(Protocol):
    spec: ModelSpec


class ProductionTrainer(ModelRunner, Protocol):
    def train_production(self, args: Any) -> ArtifactPaths:
        ...


class WalkForwardRunner(ModelRunner, Protocol):
    def run_walk_forward(self, args: Any) -> WalkForwardResult:
        ...


class BacktestRunner(ModelRunner, Protocol):
    def run_backtest(self, args: Any) -> None:
        ...


class Predictor(Protocol):
    def predict_proba(self, context: Any) -> tuple[float, float] | None:
        ...


RunnerFactory = Callable[[ModelSpec], ModelRunner]
