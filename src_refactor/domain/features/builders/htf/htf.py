from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.domain.features.primitives.indicators import (
    compute_adx,
    compute_atr,
    compute_linear_regression_slope,
    compute_rolling_vwap,
    safe_ratio_series,
)

HTF_FEATURE_COLUMNS = [
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
    "donchian_width_atr_4h",
    "donchian_width_change_4h",
    "ema_slope_4h",
    "zscore_vs_vwap_4h",
]


@dataclass(frozen=True, slots=True)
class HtfFeatureConfig:
    realized_vol_window_4h: int = 20
    range_window_4h: int = 14
    vwap_window_4h: int = 20
    ema_slope_base_window_4h: int = 21
    ema_slope_window_4h: int = 6

    @property
    def realized_vol_window(self) -> int:
        return max(2, int(self.realized_vol_window_4h))

    @property
    def range_window(self) -> int:
        return max(2, int(self.range_window_4h))

    @property
    def vwap_window(self) -> int:
        return max(2, int(self.vwap_window_4h))

    @property
    def ema_slope_base_window(self) -> int:
        return max(2, int(self.ema_slope_base_window_4h))

    @property
    def ema_slope_window(self) -> int:
        return max(2, int(self.ema_slope_window_4h))


def build_htf_frame(candles: pd.DataFrame, config: HtfFeatureConfig | None = None) -> pd.DataFrame:
    output = candles[["timestamp_ms"]].copy()
    if candles.empty:
        return pd.DataFrame(columns=["timestamp_ms", *HTF_FEATURE_COLUMNS])

    config = config or HtfFeatureConfig()
    close = pd.to_numeric(candles["close"], errors="coerce")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    volume = pd.to_numeric(candles["volume"], errors="coerce")

    for period in (1, 3, 7, 14):
        output[f"return_4h_{period}"] = np.log(close / close.shift(period))

    log_return_4h_1 = np.log(close / close.shift(1))
    output["realized_vol_4h_returns_20"] = log_return_4h_1.rolling(config.realized_vol_window).std()
    output["adx_4h"] = compute_adx(high, low, close, length=14)

    rolling_low = low.rolling(config.range_window).min()
    rolling_high = high.rolling(config.range_window).max()
    atr_14 = compute_atr(high, low, close, length=14)
    output["price_position_4h"] = safe_ratio_series(close - rolling_low, rolling_high - rolling_low)
    output["distance_to_rolling_high_4h"] = safe_ratio_series(close - rolling_high, atr_14)
    output["distance_to_rolling_low_4h"] = safe_ratio_series(close - rolling_low, atr_14)

    prev_rolling_low = rolling_low.shift(1)
    prev_rolling_high = rolling_high.shift(1)
    breakout_quality = pd.Series(0.0, index=close.index)
    upside_mask = close > prev_rolling_high
    downside_mask = close < prev_rolling_low
    breakout_quality.loc[upside_mask] = safe_ratio_series(
        close.loc[upside_mask] - prev_rolling_high.loc[upside_mask],
        atr_14.loc[upside_mask],
    )
    breakout_quality.loc[downside_mask] = -safe_ratio_series(
        prev_rolling_low.loc[downside_mask] - close.loc[downside_mask],
        atr_14.loc[downside_mask],
    )
    output["breakout_quality_4h"] = breakout_quality

    donchian_width_atr = safe_ratio_series(rolling_high - rolling_low, atr_14)
    output["donchian_width_atr_4h"] = donchian_width_atr
    output["donchian_width_change_4h"] = donchian_width_atr - donchian_width_atr.shift(3)

    ema_base = close.ewm(span=config.ema_slope_base_window, adjust=False).mean()
    output["ema_slope_4h"] = safe_ratio_series(
        compute_linear_regression_slope(ema_base, config.ema_slope_window),
        atr_14,
    )

    rolling_vwap = compute_rolling_vwap(close, high, low, volume, config.vwap_window)
    vwap_distance = close - rolling_vwap
    vwap_distance_mean = vwap_distance.rolling(config.vwap_window).mean()
    vwap_distance_std = vwap_distance.rolling(config.vwap_window).std().replace(0, np.nan)
    output["zscore_vs_vwap_4h"] = (vwap_distance - vwap_distance_mean) / vwap_distance_std

    output[HTF_FEATURE_COLUMNS] = output[HTF_FEATURE_COLUMNS].shift(1)
    return output[["timestamp_ms", *HTF_FEATURE_COLUMNS]]
