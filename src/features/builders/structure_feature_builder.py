from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, safe_ratio
from src.features.models.feature_context import FeatureContext


class StructureFeatureBuilder(FeatureBuilderContract):
    block_name = "structure"

    def provides(self) -> set[str]:
        return {
            "price_position_1h",
            "distance_to_support_1h",
            "distance_to_resistance_1h",
            "distance_to_session_high_1h",
            "distance_to_session_low_1h",
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

        if {"price_position_1h", "distance_to_support_1h", "distance_to_resistance_1h"}.intersection(active):
            rolling_low = low.rolling(24).min()
            rolling_high = high.rolling(24).max()
            if "price_position_1h" in active:
                output["price_position_1h"] = safe_ratio(close - rolling_low, rolling_high - rolling_low)

            if {"distance_to_support_1h", "distance_to_resistance_1h"}.intersection(active):
                atr_14 = context.indicator_cache.get_or_create(
                    "atr_14",
                    lambda: compute_atr(high, low, close, length=14),
                )
                if "distance_to_support_1h" in active:
                    output["distance_to_support_1h"] = safe_ratio(close - rolling_low, atr_14)
                if "distance_to_resistance_1h" in active:
                    output["distance_to_resistance_1h"] = safe_ratio(rolling_high - close, atr_14)

        if {"distance_to_session_high_1h", "distance_to_session_low_1h"}.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            session_key = frame["timestamp"].dt.floor("D")
            session_high = high.groupby(session_key).cummax()
            session_low = low.groupby(session_key).cummin()
            if "distance_to_session_high_1h" in active:
                output["distance_to_session_high_1h"] = safe_ratio(session_high - close, atr_14)
            if "distance_to_session_low_1h" in active:
                output["distance_to_session_low_1h"] = safe_ratio(close - session_low, atr_14)

        return output
