from __future__ import annotations

import pandas as pd

from src_refactor.domain.features.primitives.indicators import (
    compute_atr,
    compute_rolling_vwap,
    compute_trend_efficiency,
    safe_ratio_series,
)

STRUCTURE_FEATURE_COLUMNS = [
    "price_position_1h",
    "distance_to_support_1h",
    "distance_to_resistance_1h",
    "distance_to_session_high_1h",
    "distance_to_session_low_1h",
    "range_position_1h_48",
    "range_width_atr_1h_48",
    "range_center_distance_atr_1h_48",
    "flat_efficiency_1h_24",
    "mean_reversion_pressure_1h",
    "zscore_vs_vwap_1h",
    "bollinger_percent_b_1h_20",
    "bollinger_bandwidth_atr_1h_20",
]


def build_structure_frame(candles: pd.DataFrame) -> pd.DataFrame:
    output = candles[["timestamp_ms"]].copy()
    if candles.empty:
        return pd.DataFrame(columns=["timestamp_ms", *STRUCTURE_FEATURE_COLUMNS])

    close = pd.to_numeric(candles["close"], errors="coerce")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    volume = pd.to_numeric(candles["volume"], errors="coerce")
    atr_14 = compute_atr(high, low, close, length=14)

    rolling_low = low.rolling(24).min()
    rolling_high = high.rolling(24).max()
    output["price_position_1h"] = safe_ratio_series(close - rolling_low, rolling_high - rolling_low)
    output["distance_to_support_1h"] = safe_ratio_series(close - rolling_low, atr_14)
    output["distance_to_resistance_1h"] = safe_ratio_series(rolling_high - close, atr_14)

    timestamp = pd.to_datetime(candles["timestamp_ms"], unit="ms", utc=True)
    session_key = timestamp.dt.floor("D")
    session_high = high.groupby(session_key).cummax()
    session_low = low.groupby(session_key).cummin()
    output["distance_to_session_high_1h"] = safe_ratio_series(session_high - close, atr_14)
    output["distance_to_session_low_1h"] = safe_ratio_series(close - session_low, atr_14)

    range_low_48 = low.rolling(48).min()
    range_high_48 = high.rolling(48).max()
    range_width_48 = range_high_48 - range_low_48
    range_position_48 = safe_ratio_series(close - range_low_48, range_width_48)
    output["range_position_1h_48"] = range_position_48
    output["range_width_atr_1h_48"] = safe_ratio_series(range_width_48, atr_14)
    range_center_48 = (range_high_48 + range_low_48) / 2.0
    output["range_center_distance_atr_1h_48"] = safe_ratio_series(close - range_center_48, atr_14)

    trend_efficiency = compute_trend_efficiency(close, 24)
    output["flat_efficiency_1h_24"] = 1.0 - trend_efficiency.clip(0.0, 1.0)
    distance_from_center = (range_position_48 - 0.5) * 2.0
    output["mean_reversion_pressure_1h"] = -distance_from_center.clip(-1.0, 1.0)

    rolling_vwap = compute_rolling_vwap(close, high, low, volume, 24)
    vwap_distance = close - rolling_vwap
    vwap_distance_std = vwap_distance.rolling(24).std().replace(0, pd.NA)
    output["zscore_vs_vwap_1h"] = safe_ratio_series(vwap_distance, vwap_distance_std)

    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()
    upper_band = rolling_mean_20 + 2.0 * rolling_std_20
    lower_band = rolling_mean_20 - 2.0 * rolling_std_20
    band_width = upper_band - lower_band
    output["bollinger_percent_b_1h_20"] = safe_ratio_series(close - lower_band, band_width)
    output["bollinger_bandwidth_atr_1h_20"] = safe_ratio_series(band_width, atr_14)

    return output[["timestamp_ms", *STRUCTURE_FEATURE_COLUMNS]]
