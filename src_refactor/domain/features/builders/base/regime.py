from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.domain.features.primitives.indicators import compute_atr, safe_ratio_series

REGIME_FEATURE_COLUMNS = [
    "realized_vol_1h",
    "atr_ratio_1h",
    "volatility_regime_change_1h",
    "range_compression_1h",
    "volatility_acceleration_1h",
    "volume_24h",
    "volume_ratio_1h",
    "volume_zscore_1h",
    "dollar_volume_zscore_1h",
    "vol_of_vol_1h",
    "realized_vol_vs_ema",
    "volatility_regime_stability",
    "high_vol_stress_indicator",
    "vol_regime_classification",
]


@dataclass(frozen=True, slots=True)
class RegimeFeatureConfig:
    realized_vol_window_1h: int = 24
    range_compression_short_window_1h: int = 12
    range_compression_long_window_1h: int = 48
    volume_ratio_window_1h: int = 24
    volume_zscore_window_1h: int = 24 * 7
    volume_24h_window_1h: int = 24

    @property
    def realized_vol_window(self) -> int:
        return max(2, int(self.realized_vol_window_1h))

    @property
    def range_short_window(self) -> int:
        return max(2, int(self.range_compression_short_window_1h))

    @property
    def range_long_window(self) -> int:
        return max(self.range_short_window + 1, int(self.range_compression_long_window_1h))

    @property
    def volume_ratio_window(self) -> int:
        return max(2, int(self.volume_ratio_window_1h))

    @property
    def volume_zscore_window(self) -> int:
        return max(24, int(self.volume_zscore_window_1h))

    @property
    def volume_24h_window(self) -> int:
        return max(2, int(self.volume_24h_window_1h))


def build_regime_frame(candles: pd.DataFrame, config: RegimeFeatureConfig | None = None) -> pd.DataFrame:
    output = candles[["timestamp_ms"]].copy()
    if candles.empty:
        return pd.DataFrame(columns=["timestamp_ms", *REGIME_FEATURE_COLUMNS])

    config = config or RegimeFeatureConfig()
    close = pd.to_numeric(candles["close"], errors="coerce")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    volume = pd.to_numeric(candles["volume"], errors="coerce")

    log_return_1h_1 = np.log(close / close.shift(1))
    rvol = log_return_1h_1.rolling(config.realized_vol_window).std()
    output["realized_vol_1h"] = rvol

    atr_14 = compute_atr(high, low, close, length=14)
    atr_100 = compute_atr(high, low, close, length=100)
    atr_6 = compute_atr(high, low, close, length=6)
    atr_48 = compute_atr(high, low, close, length=48)
    regime_change = safe_ratio_series(atr_6, atr_48)
    output["atr_ratio_1h"] = safe_ratio_series(atr_14, atr_100)
    output["volatility_regime_change_1h"] = regime_change

    range_short = high.rolling(config.range_short_window).max() - low.rolling(config.range_short_window).min()
    range_long = high.rolling(config.range_long_window).max() - low.rolling(config.range_long_window).min()
    output["range_compression_1h"] = safe_ratio_series(range_short, range_long)
    output["volatility_acceleration_1h"] = regime_change - regime_change.shift(3)

    output["volume_24h"] = volume.rolling(config.volume_24h_window).sum()
    volume_mean = volume.rolling(config.volume_ratio_window).mean()
    output["volume_ratio_1h"] = safe_ratio_series(volume, volume_mean)

    volume_roll_mean = volume.rolling(config.volume_zscore_window).mean()
    volume_roll_std = volume.rolling(config.volume_zscore_window).std().replace(0, np.nan)
    dollar_volume = np.log1p((close * volume).clip(lower=0))
    dollar_roll_mean = dollar_volume.rolling(config.volume_zscore_window).mean()
    dollar_roll_std = dollar_volume.rolling(config.volume_zscore_window).std().replace(0, np.nan)
    output["volume_zscore_1h"] = (volume - volume_roll_mean) / volume_roll_std
    output["dollar_volume_zscore_1h"] = (dollar_volume - dollar_roll_mean) / dollar_roll_std

    output["vol_of_vol_1h"] = rvol.rolling(12).std()
    rvol_ema = rvol.ewm(span=48, adjust=False).mean()
    output["realized_vol_vs_ema"] = safe_ratio_series(rvol - rvol_ema, rvol_ema)

    rvol_long_mean = rvol.rolling(96).mean()
    regime_high = rvol > (rvol_long_mean * 1.2)
    regime_low = rvol < (rvol_long_mean * 0.8)
    regime_label = pd.Series(1, index=candles.index, dtype=int)
    regime_label.loc[regime_low] = 0
    regime_label.loc[regime_high] = 2
    regime_groups = (regime_label != regime_label.shift(1)).cumsum()
    regime_age = regime_label.groupby(regime_groups).cumcount() + 1
    output["volatility_regime_stability"] = regime_age.clip(0, 48) / 48.0

    atr_expanding = atr_14 > atr_48 * 1.1
    vol_high = rvol > rvol.rolling(96).quantile(0.75)
    output["high_vol_stress_indicator"] = (atr_expanding & vol_high).astype(float)

    rvol_33 = rvol.rolling(96).quantile(0.33)
    rvol_67 = rvol.rolling(96).quantile(0.67)
    output["vol_regime_classification"] = (rvol > rvol_33).astype(int) + (rvol > rvol_67).astype(int)

    return output[["timestamp_ms", *REGIME_FEATURE_COLUMNS]]
