from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import safe_ratio
from src.features.models.feature_context import FeatureContext


class VolumeFlowFeatureBuilder(FeatureBuilderContract):
    block_name = "volume_flow"

    def __init__(self) -> None:
        self.volume_pressure_window = int(getattr(cfg, "VOLUME_PRESSURE_WINDOW", 12))
        self.volume_ma_length = int(getattr(cfg, "VOLUME_MA_LENGTH", 30))

    def provides(self) -> set[str]:
        return {
            "up_volume_share_12",
            "net_candle_volume_bias_12",
            "effort_vs_result_12",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        frame = context.frame
        open_ = frame["open"]
        close = frame["close"]
        volume = frame["volume"]
        cache = context.indicator_cache

        rolling_volume_sum = volume.rolling(self.volume_pressure_window).sum().replace(0, np.nan)
        green_volume = volume.where(close > open_, 0.0)
        signed_volume = np.sign(close - open_) * volume

        log_returns = cache.get_or_create(
            "volume_flow:log_returns",
            lambda: np.log(close / close.shift(1)),
        )
        abs_return_mean = log_returns.abs().rolling(self.volume_pressure_window).mean().replace(0, np.nan)
        volume_ma = cache.get_or_create(
            f"volume_flow:volume_ma:{self.volume_ma_length}",
            lambda: volume.rolling(self.volume_ma_length, min_periods=self.volume_ma_length).mean(),
        )
        volume_ratio = safe_ratio(volume, volume_ma)
        effort = volume_ratio.rolling(self.volume_pressure_window).mean()

        feature_map = {
            "up_volume_share_12": safe_ratio(green_volume.rolling(self.volume_pressure_window).sum(), rolling_volume_sum),
            "net_candle_volume_bias_12": safe_ratio(signed_volume.rolling(self.volume_pressure_window).sum(), rolling_volume_sum),
            "effort_vs_result_12": safe_ratio(effort, abs_return_mean),
        }

        for feature_name in sorted(active):
            output[feature_name] = feature_map[feature_name]
        return output
