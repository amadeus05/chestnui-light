from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.domain.features.primitives.indicators import safe_ratio_series

INTERACTION_FEATURE_COLUMNS = [
    "vol_ratio",
    "delta_market_breadth_ema_fast_slow_1h",
    "market_breadth_ema_fast_slow_1h_zscore",
    "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
    "market_directional_pressure_1h",
    "signal_market_agreement_1h",
    "counter_market_penalty_1h",
    "trend_efficiency_24h_x_volatility_regime_change_1h",
    "trend_alignment_1h_4h",
    "breakout_quality_4h_x_volume_ratio_1h",
    "ema_fast_slow_x_vol_of_vol",
    "trend_efficiency_x_vol_stability",
    "market_pressure_x_vol_regime",
    "signal_x_high_vol_stress",
]


@dataclass(frozen=True, slots=True)
class InteractionFeatureConfig:
    market_zscore_window: int = 96

    @property
    def zscore_window(self) -> int:
        return max(10, int(self.market_zscore_window))


def build_interaction_frame(features: pd.DataFrame, config: InteractionFeatureConfig | None = None) -> pd.DataFrame:
    output = features[["timestamp_ms"]].copy()
    if features.empty:
        return pd.DataFrame(columns=["timestamp_ms", *INTERACTION_FEATURE_COLUMNS])

    config = config or InteractionFeatureConfig()
    if "realized_vol_1h" not in features.columns or "realized_vol_4h_returns_20" not in features.columns:
        raise ValueError("Feature 'vol_ratio' requires 'realized_vol_1h' and 'realized_vol_4h_returns_20'.")
    realized_vol_4h_per_hour = features["realized_vol_4h_returns_20"] / np.sqrt(4.0)
    output["vol_ratio"] = safe_ratio_series(features["realized_vol_1h"], realized_vol_4h_per_hour)

    if "market_breadth_ema_fast_slow_1h" not in features.columns:
        raise ValueError("Market breadth interaction features require 'market_breadth_ema_fast_slow_1h'.")
    breadth = pd.to_numeric(features["market_breadth_ema_fast_slow_1h"], errors="coerce")
    market_pressure = (breadth - 0.5) * 2.0
    output["delta_market_breadth_ema_fast_slow_1h"] = breadth.diff(1)
    breadth_mean = breadth.rolling(config.zscore_window).mean()
    breadth_std = breadth.rolling(config.zscore_window).std().replace(0, np.nan)
    output["market_breadth_ema_fast_slow_1h_zscore"] = (breadth - breadth_mean) / breadth_std
    output["market_directional_pressure_1h"] = market_pressure

    _require_columns(features, {"ema_fast_slow"}, "Breadth agreement features")
    signal_strength = pd.to_numeric(features["ema_fast_slow"], errors="coerce")
    signal_direction = np.sign(signal_strength)
    output["ema_fast_slow_x_market_breadth_ema_fast_slow_1h"] = signal_strength * breadth
    output["signal_market_agreement_1h"] = signal_strength * market_pressure
    disagreement = (-signal_direction * market_pressure).clip(lower=0)
    output["counter_market_penalty_1h"] = signal_strength.abs() * disagreement

    _require_columns(features, {"vol_regime_classification"}, "Feature 'market_pressure_x_vol_regime'")
    regime_normalized = (pd.to_numeric(features["vol_regime_classification"], errors="coerce") - 1.0) / 1.0
    output["market_pressure_x_vol_regime"] = market_pressure * regime_normalized

    _require_columns(
        features,
        {"trend_efficiency_24h", "volatility_regime_change_1h"},
        "Feature 'trend_efficiency_24h_x_volatility_regime_change_1h'",
    )
    output["trend_efficiency_24h_x_volatility_regime_change_1h"] = (
        features["trend_efficiency_24h"] * features["volatility_regime_change_1h"]
    )

    _require_columns(features, {"ema_fast_slow", "ema_slope_4h"}, "Feature 'trend_alignment_1h_4h'")
    output["trend_alignment_1h_4h"] = features["ema_fast_slow"] * features["ema_slope_4h"]
    _require_columns(features, {"breakout_quality_4h", "volume_ratio_1h"}, "Feature 'breakout_quality_4h_x_volume_ratio_1h'")
    output["breakout_quality_4h_x_volume_ratio_1h"] = features["breakout_quality_4h"] * features["volume_ratio_1h"]

    output["ema_fast_slow_x_vol_of_vol"] = (
        features["ema_fast_slow"] * features["vol_of_vol_1h"]
        if "ema_fast_slow" in features.columns and "vol_of_vol_1h" in features.columns
        else np.nan
    )
    output["trend_efficiency_x_vol_stability"] = (
        features["trend_efficiency_24h"] * features["volatility_regime_stability"]
        if "trend_efficiency_24h" in features.columns and "volatility_regime_stability" in features.columns
        else np.nan
    )
    output["signal_x_high_vol_stress"] = (
        features["ema_fast_slow"] * features["high_vol_stress_indicator"]
        if "ema_fast_slow" in features.columns and "high_vol_stress_indicator" in features.columns
        else np.nan
    )
    return output[["timestamp_ms", *INTERACTION_FEATURE_COLUMNS]]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if required.issubset(frame.columns):
        return
    missing = ", ".join(sorted(required - set(frame.columns)))
    raise ValueError(f"{label} requires: {missing}")
