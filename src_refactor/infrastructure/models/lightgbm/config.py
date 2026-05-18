from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LightGbmTrainingConfig:
    seed: int = 42
    n_estimators: int = 800
    learning_rate: float = 0.005
    num_leaves: int = 15
    min_child_samples: int = 150
    max_depth: int = 5
    subsample: float = 0.6
    colsample_bytree: float = 0.5
    reg_alpha: float = 1.0
    reg_lambda: float = 3.0
    class_weight: str | None = "balanced"
    min_split_gain: float = 0.01
    subsample_freq: int = 1
    use_symbol_feature: bool = True
    enable_feature_clip: bool = False
    feature_clip_lower_q: float = 0.01
    feature_clip_upper_q: float = 0.99
    directional_proba_threshold: float = 0.5
    min_signal_gap: float = 0.0
    disabled_feature_columns: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, payload: dict[str, Any] | None) -> "LightGbmTrainingConfig":
        if not payload:
            return cls()
        defaults = cls()
        return cls(
            seed=int(payload.get("seed", defaults.seed)),
            n_estimators=int(payload.get("n_estimators", defaults.n_estimators)),
            learning_rate=float(payload.get("learning_rate", defaults.learning_rate)),
            num_leaves=int(payload.get("num_leaves", defaults.num_leaves)),
            min_child_samples=int(payload.get("min_child_samples", defaults.min_child_samples)),
            max_depth=int(payload.get("max_depth", defaults.max_depth)),
            subsample=float(payload.get("subsample", defaults.subsample)),
            colsample_bytree=float(payload.get("colsample_bytree", defaults.colsample_bytree)),
            reg_alpha=float(payload.get("reg_alpha", defaults.reg_alpha)),
            reg_lambda=float(payload.get("reg_lambda", defaults.reg_lambda)),
            class_weight=payload.get("class_weight", defaults.class_weight),
            min_split_gain=float(payload.get("min_split_gain", defaults.min_split_gain)),
            subsample_freq=int(payload.get("subsample_freq", defaults.subsample_freq)),
            use_symbol_feature=bool(payload.get("use_symbol_feature", defaults.use_symbol_feature)),
            enable_feature_clip=bool(payload.get("enable_feature_clip", defaults.enable_feature_clip)),
            feature_clip_lower_q=float(payload.get("feature_clip_lower_q", defaults.feature_clip_lower_q)),
            feature_clip_upper_q=float(payload.get("feature_clip_upper_q", defaults.feature_clip_upper_q)),
            directional_proba_threshold=float(
                payload.get("directional_proba_threshold", defaults.directional_proba_threshold)
            ),
            min_signal_gap=float(payload.get("min_signal_gap", defaults.min_signal_gap)),
            disabled_feature_columns=tuple(str(value) for value in payload.get("disabled_feature_columns", ())),
            metadata=dict(payload),
        )

    def model_params(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "class_weight": self.class_weight,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": -1,
            "min_split_gain": self.min_split_gain,
            "subsample_freq": self.subsample_freq,
        }
