from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_adx, compute_trend_efficiency, safe_ratio
from src.features.models.feature_context import FeatureContext


class HtfContextFeatureBuilder(FeatureBuilderContract):
    block_name = "htf_context"

    def provides(self) -> set[str]:
        return {
            "htf_return_1",
            "htf_return_3",
            "htf_ema_fast_slow",
            "htf_ema_slope",
            "htf_adx_14",
            "htf_realized_vol_20",
            "htf_price_position_24",
            "htf_trend_efficiency_24",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        frame = context.frame
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        cache = context.indicator_cache

        ema_fast = cache.get_or_create(
            "htf_context:ema_fast",
            lambda: close.ewm(span=12, adjust=False).mean(),
        )
        ema_slow = cache.get_or_create(
            "htf_context:ema_slow",
            lambda: close.ewm(span=48, adjust=False).mean(),
        )
        log_returns = cache.get_or_create(
            "htf_context:log_returns",
            lambda: np.log(close / close.shift(1)),
        )

        if "htf_return_1" in active:
            output["htf_return_1"] = np.log(close / close.shift(1))
        if "htf_return_3" in active:
            output["htf_return_3"] = np.log(close / close.shift(3))
        if "htf_ema_fast_slow" in active:
            output["htf_ema_fast_slow"] = safe_ratio(ema_fast - ema_slow, ema_slow)
        if "htf_ema_slope" in active:
            output["htf_ema_slope"] = safe_ratio(ema_fast.diff(2), close)
        if "htf_adx_14" in active:
            output["htf_adx_14"] = compute_adx(high, low, close, length=14)
        if "htf_realized_vol_20" in active:
            output["htf_realized_vol_20"] = log_returns.rolling(20).std()
        if "htf_price_position_24" in active:
            rolling_low = low.rolling(24).min()
            rolling_high = high.rolling(24).max()
            output["htf_price_position_24"] = safe_ratio(
                close - rolling_low,
                (rolling_high - rolling_low).replace(0, np.nan),
            )
        if "htf_trend_efficiency_24" in active:
            output["htf_trend_efficiency_24"] = compute_trend_efficiency(close, 24)
        return output
