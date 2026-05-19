from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

ModelType: TypeAlias = Literal["lightgbm", "lstm_features", "lstm_candles", "ensemble"]
ModelVariantId: TypeAlias = str
ModelInput: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class LightGbmInput:
    features: pd.DataFrame
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LstmFeatureSequenceInput:
    sequence: np.ndarray
    feature_names: tuple[str, ...]
    window_size: int
    targets: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LstmCandleWindowInput:
    candles: np.ndarray
    candle_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    window_size: int = 0
    targets: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_type: ModelType
    timeframe: str
    profile: str = "baseline"
    symbols: tuple[str, ...] = ()
    input_profile: str = "default"
    artifact_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> ModelVariantId:
        return f"{self.model_type}__{self.timeframe}__{self.profile}"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    spec: ModelSpec
    uri: str | Path
    fold_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prediction:
    timestamp: pd.Timestamp
    symbol: str
    timeframe: str
    model_id: ModelVariantId
    direction: int
    confidence: float
    fold_id: int | None = None
    proba_long: float | None = None
    proba_short: float | None = None
    signal_gap: float | None = None
    stop_pct: float | None = None
    take_pct: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("Prediction.direction must be one of -1, 0, 1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Prediction.confidence must be in [0, 1].")
