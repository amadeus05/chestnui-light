from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import safe_ratio
from src.features.models.feature_context import FeatureContext


class InteractionFeatureBuilder(FeatureBuilderContract):
    block_name = "interactions"

    def provides(self) -> set[str]:
        return {
            "vol_ratio",
            "delta_market_breadth_ema_fast_slow_1h",
            "market_breadth_ema_fast_slow_1h_zscore",
            "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
            "market_directional_pressure_1h",
            "signal_market_agreement_1h",
            "counter_market_penalty_1h",
            "trend_efficiency_24h_x_volatility_regime_change_1h",
            "trend_alignment_1h_4h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "vol_ratio" in active:
            if "realized_vol_1h" not in frame.columns or "realized_vol_4h_returns_20" not in frame.columns:
                raise ValueError("Feature 'vol_ratio' requires 'realized_vol_1h' and 'realized_vol_4h_returns_20'.")
            realized_vol_4h_per_hour = frame["realized_vol_4h_returns_20"] / np.sqrt(4.0)
            output["vol_ratio"] = safe_ratio(frame["realized_vol_1h"], realized_vol_4h_per_hour)

        breadth_request = {
            "delta_market_breadth_ema_fast_slow_1h",
            "market_breadth_ema_fast_slow_1h_zscore",
            "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
            "market_directional_pressure_1h",
            "signal_market_agreement_1h",
            "counter_market_penalty_1h",
        }
        if breadth_request.intersection(active):
            if "market_breadth_ema_fast_slow_1h" not in frame.columns:
                raise ValueError("Market breadth interaction features require 'market_breadth_ema_fast_slow_1h'.")
            breadth = frame["market_breadth_ema_fast_slow_1h"]
            market_pressure = (breadth - 0.5) * 2.0
            if "delta_market_breadth_ema_fast_slow_1h" in active:
                output["delta_market_breadth_ema_fast_slow_1h"] = breadth.diff(1)
            if "market_breadth_ema_fast_slow_1h_zscore" in active:
                zscore_window = max(10, int(getattr(cfg, "MARKET_ZSCORE_WINDOW", 96)))
                breadth_mean = breadth.rolling(zscore_window).mean()
                breadth_std = breadth.rolling(zscore_window).std().replace(0, np.nan)
                output["market_breadth_ema_fast_slow_1h_zscore"] = (breadth - breadth_mean) / breadth_std
            if "market_directional_pressure_1h" in active:
                output["market_directional_pressure_1h"] = market_pressure
            if "ema_fast_slow_x_market_breadth_ema_fast_slow_1h" in active:
                if "ema_fast_slow" not in frame.columns:
                    raise ValueError(
                        "Feature 'ema_fast_slow_x_market_breadth_ema_fast_slow_1h' requires 'ema_fast_slow'."
                    )
                output["ema_fast_slow_x_market_breadth_ema_fast_slow_1h"] = frame["ema_fast_slow"] * breadth
            if {"signal_market_agreement_1h", "counter_market_penalty_1h"}.intersection(active):
                if "ema_fast_slow" not in frame.columns:
                    raise ValueError(
                        "Breadth agreement features require 'ema_fast_slow'."
                    )
                signal_strength = pd.to_numeric(frame["ema_fast_slow"], errors="coerce")
                signal_direction = np.sign(signal_strength)
                if "signal_market_agreement_1h" in active:
                    output["signal_market_agreement_1h"] = signal_strength * market_pressure
                if "counter_market_penalty_1h" in active:
                    disagreement = (-signal_direction * market_pressure).clip(lower=0)
                    output["counter_market_penalty_1h"] = signal_strength.abs() * disagreement

        if "trend_efficiency_24h_x_volatility_regime_change_1h" in active:
            required = {"trend_efficiency_24h", "volatility_regime_change_1h"}
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError(
                    "Feature 'trend_efficiency_24h_x_volatility_regime_change_1h' requires: " + missing
                )
            output["trend_efficiency_24h_x_volatility_regime_change_1h"] = (
                frame["trend_efficiency_24h"] * frame["volatility_regime_change_1h"]
            )

        if "trend_alignment_1h_4h" in active:
            required = {"ema_fast_slow", "ema_slope_4h"}
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError("Feature 'trend_alignment_1h_4h' requires: " + missing)
            output["trend_alignment_1h_4h"] = frame["ema_fast_slow"] * frame["ema_slope_4h"]

        return output
