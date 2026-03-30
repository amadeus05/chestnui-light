from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, compute_linear_regression_slope, compute_trend_efficiency, safe_ratio
from src.features.models.feature_context import FeatureContext


class MomentumFeatureBuilder(FeatureBuilderContract):
    block_name = "momentum"

    def provides(self) -> set[str]:
        return {
            "return_1h_6",
            "return_1h_12",
            "return_1h_24",
            "ema_fast_slow",
            "linear_regression_slope_atr_1h_12",
            "linear_regression_slope_atr_1h_24",
            "trend_persistence_score_12",
            "trend_persistence_score_24",
            "trend_efficiency_24h",
            "slope_acceleration_1h_12_24",
            "ema_slope_acceleration_1h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = frame["close"]
        ema_fast_window = max(2, int(getattr(cfg, "EMA_FAST_WINDOW", 12)))
        ema_slow_window = max(ema_fast_window + 1, int(getattr(cfg, "EMA_SLOW_WINDOW", 48)))

        if {"return_1h_6", "return_1h_12", "return_1h_24"}.intersection(active):
            for period in (6, 12, 24):
                feature_name = f"return_1h_{period}"
                if feature_name in active:
                    output[feature_name] = np.log(close / close.shift(period))

        ema_fast_slow = None
        if {"ema_fast_slow", "ema_slope_acceleration_1h"}.intersection(active):
            ema_fast = context.indicator_cache.get_or_create(
                "ema_fast",
                lambda: close.ewm(span=ema_fast_window, adjust=False).mean(),
            )
            ema_slow = context.indicator_cache.get_or_create(
                "ema_slow",
                lambda: close.ewm(span=ema_slow_window, adjust=False).mean(),
            )
            ema_fast_slow = context.indicator_cache.get_or_create(
                "ema_fast_slow",
                lambda: safe_ratio(ema_fast - ema_slow, ema_slow),
            )
            if "ema_fast_slow" in active:
                output["ema_fast_slow"] = ema_fast_slow

        slope_request = {
            "linear_regression_slope_atr_1h_12",
            "linear_regression_slope_atr_1h_24",
            "slope_acceleration_1h_12_24",
        }
        if slope_request.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(frame["high"], frame["low"], close, length=14),
            )
            slope_12 = None
            slope_24 = None
            if {"linear_regression_slope_atr_1h_12", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_12 = context.indicator_cache.get_or_create(
                    "lr_close_12",
                    lambda: compute_linear_regression_slope(close, 12),
                )
            if {"linear_regression_slope_atr_1h_24", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_24 = context.indicator_cache.get_or_create(
                    "lr_close_24",
                    lambda: compute_linear_regression_slope(close, 24),
                )
            if "linear_regression_slope_atr_1h_12" in active and slope_12 is not None:
                output["linear_regression_slope_atr_1h_12"] = safe_ratio(slope_12, atr_14)
            if "linear_regression_slope_atr_1h_24" in active and slope_24 is not None:
                output["linear_regression_slope_atr_1h_24"] = safe_ratio(slope_24, atr_14)
            if "slope_acceleration_1h_12_24" in active and slope_12 is not None and slope_24 is not None:
                output["slope_acceleration_1h_12_24"] = safe_ratio(slope_12 - slope_24, atr_14)

        if {"trend_persistence_score_12", "trend_persistence_score_24"}.intersection(active):
            signed_step = context.indicator_cache.get_or_create(
                "signed_step",
                lambda: pd.Series(np.sign(close.diff()), index=close.index),
            )
            if "trend_persistence_score_12" in active:
                output["trend_persistence_score_12"] = signed_step.rolling(12).mean()
            if "trend_persistence_score_24" in active:
                output["trend_persistence_score_24"] = signed_step.rolling(24).mean()

        if "trend_efficiency_24h" in active:
            output["trend_efficiency_24h"] = context.indicator_cache.get_or_create(
                "trend_efficiency_24h",
                lambda: compute_trend_efficiency(close, 24),
            )

        if "ema_slope_acceleration_1h" in active:
            if ema_fast_slow is None:
                raise ValueError("Feature 'ema_slope_acceleration_1h' requires 'ema_fast_slow'.")
            output["ema_slope_acceleration_1h"] = ema_fast_slow - ema_fast_slow.shift(3)

        return output
