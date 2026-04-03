from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_adx, compute_atr, compute_linear_regression_slope, compute_rolling_vwap, safe_ratio
from src.features.models.feature_context import FeatureContext


class HtfFeatureBuilder(FeatureBuilderContract):
    block_name = "htf"

    def provides(self) -> set[str]:
        return {
            "return_4h_1",
            "return_4h_3",
            "return_4h_7",
            "return_4h_14",
            "realized_vol_4h_returns_20",
            "adx_4h",
            "price_position_4h",
            "distance_to_rolling_high_4h",
            "distance_to_rolling_low_4h",
            "breakout_quality_4h",
            "ema_slope_4h",
            "zscore_vs_vwap_4h",
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
        realized_vol_window = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_4H", 20)))
        range_window = max(2, int(getattr(cfg, "RANGE_WINDOW_4H", 14)))
        vwap_window = max(2, int(getattr(cfg, "VWAP_WINDOW_4H", 20)))
        ema_slope_base_window = max(2, int(getattr(cfg, "EMA_SLOPE_BASE_WINDOW_4H", 21)))
        ema_slope_window = max(2, int(getattr(cfg, "EMA_SLOPE_WINDOW_4H", 6)))

        if {"return_4h_1", "return_4h_3", "return_4h_7", "return_4h_14"}.intersection(active):
            for period in (1, 3, 7, 14):
                feature_name = f"return_4h_{period}"
                if feature_name in active:
                    output[feature_name] = np.log(close / close.shift(period))

        if "realized_vol_4h_returns_20" in active:
            log_return_4h_1 = np.log(close / close.shift(1))
            output["realized_vol_4h_returns_20"] = log_return_4h_1.rolling(realized_vol_window).std()

        if "adx_4h" in active:
            output["adx_4h"] = compute_adx(high, low, close, length=14)

        structure_request = {
            "price_position_4h",
            "distance_to_rolling_high_4h",
            "distance_to_rolling_low_4h",
            "breakout_quality_4h",
            "ema_slope_4h",
        }
        if structure_request.intersection(active):
            rolling_low = low.rolling(range_window).min()
            rolling_high = high.rolling(range_window).max()
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            if "price_position_4h" in active:
                output["price_position_4h"] = safe_ratio(close - rolling_low, rolling_high - rolling_low)
            if "distance_to_rolling_high_4h" in active:
                output["distance_to_rolling_high_4h"] = safe_ratio(close - rolling_high, atr_14)
            if "distance_to_rolling_low_4h" in active:
                output["distance_to_rolling_low_4h"] = safe_ratio(close - rolling_low, atr_14)
            if "breakout_quality_4h" in active:
                prev_rolling_low = rolling_low.shift(1)
                prev_rolling_high = rolling_high.shift(1)
                breakout_quality = pd.Series(0.0, index=close.index)
                upside_mask = close > prev_rolling_high
                downside_mask = close < prev_rolling_low
                breakout_quality.loc[upside_mask] = safe_ratio(
                    close.loc[upside_mask] - prev_rolling_high.loc[upside_mask],
                    atr_14.loc[upside_mask],
                )
                breakout_quality.loc[downside_mask] = -safe_ratio(
                    prev_rolling_low.loc[downside_mask] - close.loc[downside_mask],
                    atr_14.loc[downside_mask],
                )
                output["breakout_quality_4h"] = breakout_quality
            if "ema_slope_4h" in active:
                ema_base = context.indicator_cache.get_or_create(
                    "ema_base_4h",
                    lambda: close.ewm(span=ema_slope_base_window, adjust=False).mean(),
                )
                output["ema_slope_4h"] = safe_ratio(
                    compute_linear_regression_slope(ema_base, ema_slope_window),
                    atr_14,
                )

        if "zscore_vs_vwap_4h" in active:
            rolling_vwap = compute_rolling_vwap(close, high, low, frame["volume"], vwap_window)
            vwap_distance = close - rolling_vwap
            vwap_distance_mean = vwap_distance.rolling(vwap_window).mean()
            vwap_distance_std = vwap_distance.rolling(vwap_window).std().replace(0, np.nan)
            output["zscore_vs_vwap_4h"] = (vwap_distance - vwap_distance_mean) / vwap_distance_std

        feature_columns = [column for column in output.columns if column != "timestamp"]
        output[feature_columns] = output[feature_columns].shift(1)
        return output
