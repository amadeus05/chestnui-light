from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, safe_ratio
from src.features.models.feature_context import FeatureContext


class RegimeFeatureBuilder(FeatureBuilderContract):
    block_name = "regime"

    def provides(self) -> set[str]:
        return {
            "realized_vol_1h",
            "atr_ratio_1h",
            "volatility_regime_change_1h",
            "range_compression_1h",
            "volatility_acceleration_1h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        realized_vol_window = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_1H", 24)))
        range_short_window = max(2, int(getattr(cfg, "RANGE_COMPRESSION_SHORT_WINDOW_1H", 12)))
        range_long_window = max(range_short_window + 1, int(getattr(cfg, "RANGE_COMPRESSION_LONG_WINDOW_1H", 48)))

        if "realized_vol_1h" in active:
            log_return_1h_1 = np.log(close / close.shift(1))
            output["realized_vol_1h"] = log_return_1h_1.rolling(realized_vol_window).std()

        if {"atr_ratio_1h", "volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            if "atr_ratio_1h" in active:
                atr_100 = context.indicator_cache.get_or_create(
                    "atr_100",
                    lambda: compute_atr(high, low, close, length=100),
                )
                output["atr_ratio_1h"] = safe_ratio(atr_14, atr_100)

            if {"volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
                atr_6 = context.indicator_cache.get_or_create(
                    "atr_6",
                    lambda: compute_atr(high, low, close, length=6),
                )
                atr_48 = context.indicator_cache.get_or_create(
                    "atr_48",
                    lambda: compute_atr(high, low, close, length=48),
                )
                regime_change = context.indicator_cache.get_or_create(
                    "volatility_regime_change_1h",
                    lambda: safe_ratio(atr_6, atr_48),
                )
                if "volatility_regime_change_1h" in active:
                    output["volatility_regime_change_1h"] = regime_change
                if "volatility_acceleration_1h" in active:
                    output["volatility_acceleration_1h"] = regime_change - regime_change.shift(3)

        if "range_compression_1h" in active:
            range_short = high.rolling(range_short_window).max() - low.rolling(range_short_window).min()
            range_long = high.rolling(range_long_window).max() - low.rolling(range_long_window).min()
            output["range_compression_1h"] = safe_ratio(range_short, range_long)

        return output
