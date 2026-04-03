from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class FundingFeatureBuilder(FeatureBuilderContract):
    block_name = "funding"

    def provides(self) -> set[str]:
        return {
            "funding_rate_8h",
            "funding_rate_zscore_7d",
            "funding_rate_change_24h",
            "longs_overheated_1h",
            "shorts_overheated_1h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "funding_rate" not in frame.columns:
            raise ValueError("Funding features require raw column 'funding_rate'.")

        funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
        zscore_window = max(24, int(getattr(cfg, "FUNDING_ZSCORE_WINDOW_1H", 24 * 7)))
        change_lookback = max(1, int(getattr(cfg, "FUNDING_CHANGE_LOOKBACK_1H", 24)))

        funding_zscore = None
        if {"funding_rate_zscore_7d", "longs_overheated_1h", "shorts_overheated_1h"}.intersection(active):
            rolling_mean = funding_rate.rolling(zscore_window).mean()
            rolling_std = funding_rate.rolling(zscore_window).std().replace(0, np.nan)
            funding_zscore = (funding_rate - rolling_mean) / rolling_std

        if "funding_rate_8h" in active:
            output["funding_rate_8h"] = funding_rate
        if "funding_rate_zscore_7d" in active and funding_zscore is not None:
            output["funding_rate_zscore_7d"] = funding_zscore
        if "funding_rate_change_24h" in active:
            output["funding_rate_change_24h"] = funding_rate - funding_rate.shift(change_lookback)
        if "longs_overheated_1h" in active and funding_zscore is not None:
            output["longs_overheated_1h"] = funding_zscore.clip(lower=0)
        if "shorts_overheated_1h" in active and funding_zscore is not None:
            output["shorts_overheated_1h"] = (-funding_zscore).clip(lower=0)

        return output
