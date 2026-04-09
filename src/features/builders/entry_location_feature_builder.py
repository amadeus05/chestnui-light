from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, compute_trend_efficiency, safe_ratio
from src.features.models.feature_context import FeatureContext


class EntryLocationFeatureBuilder(FeatureBuilderContract):
    block_name = "entry_location"

    def __init__(self) -> None:
        self.atr_length = int(getattr(cfg, "CANDLE_STRUCTURE_ATR_LENGTH", 14))
        self.efficiency_window = int(getattr(cfg, "CANDLE_STRUCTURE_EFFICIENCY_WINDOW", 12))

    def provides(self) -> set[str]:
        return {
            "close_location_in_bar",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "true_range_to_atr_14",
            "efficiency_ratio_12",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        frame = context.frame
        open_ = frame["open"]
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        cache = context.indicator_cache

        candle_range = (high - low).replace(0, np.nan)
        candle_body_high = pd.concat([open_, close], axis=1).max(axis=1)
        candle_body_low = pd.concat([open_, close], axis=1).min(axis=1)
        upper_wick = (high - candle_body_high).clip(lower=0.0)
        lower_wick = (candle_body_low - low).clip(lower=0.0)

        atr_14 = cache.get_or_create(
            f"entry_location:atr:{self.atr_length}",
            lambda: compute_atr(high, low, close, length=self.atr_length),
        )
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        efficiency_ratio = cache.get_or_create(
            f"entry_location:efficiency:{self.efficiency_window}",
            lambda: compute_trend_efficiency(close, self.efficiency_window),
        )

        feature_map = {
            "close_location_in_bar": safe_ratio(close - low, candle_range),
            "upper_wick_ratio": safe_ratio(upper_wick, candle_range),
            "lower_wick_ratio": safe_ratio(lower_wick, candle_range),
            "true_range_to_atr_14": safe_ratio(true_range, atr_14),
            "efficiency_ratio_12": efficiency_ratio,
        }

        for feature_name in sorted(active):
            output[feature_name] = feature_map[feature_name]
        return output
