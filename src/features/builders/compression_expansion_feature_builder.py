from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_donchian_channels, safe_ratio
from src.features.models.feature_context import FeatureContext


class CompressionExpansionFeatureBuilder(FeatureBuilderContract):
    block_name = "compression_expansion"

    def __init__(self) -> None:
        self.donchian_length = int(getattr(cfg, "DONCHIAN_LENGTH", 96))
        self.width_zscore_lookback = int(getattr(cfg, "COMPRESSION_WIDTH_ZSCORE_LOOKBACK", 288))
        self.short_vol_window = int(getattr(cfg, "COMPRESSION_SHORT_VOL_WINDOW", 12))
        self.long_vol_window = int(getattr(cfg, "COMPRESSION_LONG_VOL_WINDOW", 96))
        self.change_bars = int(getattr(cfg, "COMPRESSION_CHANGE_BARS", 3))
        self.range_fast_window = int(getattr(cfg, "COMPRESSION_RANGE_FAST_WINDOW", 3))
        self.range_slow_window = int(getattr(cfg, "COMPRESSION_RANGE_SLOW_WINDOW", 12))
        self.volume_ma_length = int(getattr(cfg, "VOLUME_MA_LENGTH", 30))

    def provides(self) -> set[str]:
        return {
            "donchian_width_zscore_96_288",
            "donchian_width_change_3",
            "donchian_width_acceleration_3",
            "realized_vol_ratio_12_96",
            "realized_vol_change_3",
            "range_expansion_ratio_3_12",
            "range_expansion_change_3",
            "volume_ratio_change_3",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        frame = context.frame
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        volume = frame["volume"]
        cache = context.indicator_cache

        donchian_upper, _, donchian_lower = cache.get_or_create(
            f"compression_expansion:donchian:{self.donchian_length}",
            lambda: compute_donchian_channels(high, low, length=self.donchian_length, shift=1),
        )
        width_pct = safe_ratio((donchian_upper - donchian_lower).replace(0, np.nan), close).abs()
        width_mean = width_pct.rolling(self.width_zscore_lookback).mean()
        width_std = width_pct.rolling(self.width_zscore_lookback).std().replace(0, np.nan)
        width_change = width_pct - width_pct.shift(self.change_bars)

        log_returns = cache.get_or_create(
            "compression_expansion:log_returns",
            lambda: np.log(close / close.shift(1)),
        )
        realized_vol_fast = log_returns.rolling(self.short_vol_window).std()
        realized_vol_slow = log_returns.rolling(self.long_vol_window).std()

        candle_range_pct = safe_ratio((high - low), close).abs()
        range_fast = candle_range_pct.rolling(self.range_fast_window).mean()
        range_slow = candle_range_pct.rolling(self.range_slow_window).mean()

        volume_ma = cache.get_or_create(
            f"compression_expansion:volume_ma:{self.volume_ma_length}",
            lambda: volume.rolling(self.volume_ma_length, min_periods=self.volume_ma_length).mean(),
        )
        volume_ratio = safe_ratio(volume, volume_ma)
        volume_ratio_change = volume_ratio - volume_ratio.shift(self.change_bars)

        feature_map = {
            "donchian_width_zscore_96_288": safe_ratio(width_pct - width_mean, width_std),
            "donchian_width_change_3": width_change,
            "donchian_width_acceleration_3": width_change - width_change.shift(self.change_bars),
            "realized_vol_ratio_12_96": safe_ratio(realized_vol_fast, realized_vol_slow),
            "realized_vol_change_3": realized_vol_fast - realized_vol_fast.shift(self.change_bars),
            "range_expansion_ratio_3_12": safe_ratio(range_fast, range_slow),
            "range_expansion_change_3": safe_ratio(range_fast, range_slow) - safe_ratio(range_fast, range_slow).shift(self.change_bars),
            "volume_ratio_change_3": volume_ratio_change,
        }

        for feature_name in sorted(active):
            output[feature_name] = feature_map[feature_name]
        return output
