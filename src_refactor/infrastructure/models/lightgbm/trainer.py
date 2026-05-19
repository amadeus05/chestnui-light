from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelArtifactStore, ModelPredictor, ModelTrainer
from src_refactor.core.types import ModelArtifact, ModelInput, ModelSpec, WalkForwardFold
from src_refactor.core.types import LightGbmInput
from src_refactor.infrastructure.models.lightgbm.config import LightGbmTrainingConfig
from src_refactor.infrastructure.models.lightgbm.predictor import LightGbmPredictor

LABEL_TO_CLASS = {-1: 0, 1: 1}
TARGET_COLUMN = "Target"
SYMBOL_COLUMN = "symbol"


@dataclass(frozen=True, slots=True)
class LightGbmTrainer(ModelTrainer):
    def train(
        self,
        train_input: ModelInput,
        config: ExperimentConfig,
        fold: WalkForwardFold | None = None,
    ) -> ModelArtifact:
        if not isinstance(train_input, LightGbmInput):
            raise TypeError("LightGbmTrainer expects LightGbmInput.")
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is required to train LightGBM models.") from exc

        lightgbm_config = train_input.metadata.get("config")
        if not isinstance(lightgbm_config, LightGbmTrainingConfig):
            lightgbm_config = LightGbmTrainingConfig.from_metadata(config.model.metadata)

        frame = train_input.metadata.get("frame")
        if not isinstance(frame, pd.DataFrame) or TARGET_COLUMN not in frame.columns:
            raise ValueError("LightGBM training requires a frame with Target column in input metadata.")

        target = frame[TARGET_COLUMN].map(LABEL_TO_CLASS)
        valid_mask = target.notna()
        x_train = train_input.features.loc[valid_mask].copy()
        y_train = target.loc[valid_mask].astype(int)
        w_train = compute_sample_weights(frame.loc[valid_mask], lightgbm_config)

        model = lgb.LGBMClassifier(**lightgbm_config.model_params())
        categorical_feature = [SYMBOL_COLUMN] if SYMBOL_COLUMN in train_input.feature_names else "auto"
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            categorical_feature=categorical_feature,
        )

        artifact_uri = _artifact_dir(config.model, fold) / "model.joblib"
        metadata = {
            "feature_columns": list(train_input.feature_names),
            "feature_clip": {"bounds": train_input.metadata.get("clip_bounds", {})},
            "label_mapping": {"short": 0, "long": 1},
            "inverse_label_mapping": {"0": -1, "1": 1},
            "symbols": list(config.symbols or config.model.symbols),
            "rows": int(len(x_train)),
            "directional_proba_threshold": lightgbm_config.directional_proba_threshold,
            "min_signal_gap": lightgbm_config.min_signal_gap,
            "training": lightgbm_config.metadata,
            **_runtime_metadata(config.model.metadata),
        }
        return ModelArtifact(
            spec=config.model,
            uri=artifact_uri,
            fold_id=fold.fold_id if fold is not None else None,
            metadata={"model": model, **metadata},
        )


@dataclass(frozen=True, slots=True)
class LightGbmArtifactStore(ModelArtifactStore):
    root: Path = Path("models")

    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        artifact_path = Path(artifact.uri)
        if not artifact_path.is_absolute():
            artifact_path = self.root / artifact_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        model = artifact.metadata.get("model")
        if model is None:
            raise ValueError("LightGBM artifact metadata must contain trained model.")
        joblib.dump(model, artifact_path)

        metadata = {key: value for key, value in artifact.metadata.items() if key != "model"}
        metadata_path = artifact_path.with_name("features.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return ModelArtifact(
            spec=artifact.spec,
            uri=artifact_path,
            fold_id=artifact.fold_id,
            metadata=metadata,
        )

    def load_predictor(self, spec: ModelSpec, fold_id: int | None = None) -> ModelPredictor:
        artifact_path = Path(spec.artifact_uri) if spec.artifact_uri else self.root / _artifact_dir(spec, fold_id) / "model.joblib"
        metadata_path = artifact_path.with_name("features.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        effective_spec = ModelSpec(
            model_type=spec.model_type,
            timeframe=spec.timeframe,
            profile=spec.profile,
            symbols=spec.symbols,
            input_profile=spec.input_profile,
            artifact_uri=str(artifact_path),
            metadata={**metadata, **spec.metadata},
        )
        return LightGbmPredictor(spec=effective_spec, model=joblib.load(artifact_path))


def compute_sample_weights(frame: pd.DataFrame, config: LightGbmTrainingConfig) -> np.ndarray:
    timestamps = pd.to_datetime(frame["timestamp"])
    half_life_days = float(config.metadata.get("sample_weight_half_life_days", 90.0))
    days_ago = (timestamps.max() - timestamps).dt.total_seconds() / 86400.0
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago.to_numpy())

    if bool(config.metadata.get("regime_aware_weighting", True)):
        recent_days = float(config.metadata.get("regime_recent_days_boost", 30.0))
        boost_factor = float(config.metadata.get("regime_recent_boost_factor", 2.0))
        weights = np.where(days_ago <= recent_days, weights * boost_factor, weights)

    min_weight = float(config.metadata.get("sample_weight_min", 0.8))
    max_weight = float(config.metadata.get("sample_weight_max", 1.35))
    if max_weight < min_weight:
        max_weight = min_weight
    return np.clip(weights, min_weight, max_weight)


def _artifact_dir(spec: ModelSpec, fold: WalkForwardFold | int | None) -> Path:
    base = Path(spec.model_type) / spec.timeframe / spec.profile
    fold_id = fold.fold_id if isinstance(fold, WalkForwardFold) else fold
    return base / f"fold_{fold_id}" if fold_id is not None else base


def _runtime_metadata(model_metadata: dict) -> dict[str, object]:
    return {"labeling": model_metadata["labeling"]} if "labeling" in model_metadata else {}
