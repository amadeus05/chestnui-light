from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.domain.features.primitives.indicators import (
    compute_atr,
    compute_linear_regression_slope,
    compute_rsi,
    compute_trend_efficiency,
    safe_ratio_series,
)

MOMENTUM_FEATURE_COLUMNS = [
    "return_1h_6",
    "return_1h_12",
    "return_1h_24",
    "ema_fast_slow",
    "rsi_1h",
    "linear_regression_slope_atr_1h_12",
    "linear_regression_slope_atr_1h_24",
    "trend_persistence_score_12",
    "trend_persistence_score_24",
    "trend_efficiency_24h",
    "slope_acceleration_1h_12_24",
    "ema_slope_acceleration_1h",
]


@dataclass(frozen=True, slots=True)
class MomentumFeatureConfig:
    ema_fast_window: int = 12
    ema_slow_window: int = 48
    rsi_length: int = 14

    @property
    def fast_window(self) -> int:
        return max(2, int(self.ema_fast_window))

    @property
    def slow_window(self) -> int:
        return max(self.fast_window + 1, int(self.ema_slow_window))

    @property
    def normalized_rsi_length(self) -> int:
        return max(2, int(self.rsi_length))


def build_momentum_frame(candles: pd.DataFrame, config: MomentumFeatureConfig | None = None) -> pd.DataFrame:
    output = candles[["timestamp_ms"]].copy()
    if candles.empty:
        return pd.DataFrame(columns=["timestamp_ms", *MOMENTUM_FEATURE_COLUMNS])

    config = config or MomentumFeatureConfig()
    close = pd.to_numeric(candles["close"], errors="coerce")

    for period in (6, 12, 24):
        output[f"return_1h_{period}"] = np.log(close / close.shift(period))

    ema_fast = close.ewm(span=config.fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=config.slow_window, adjust=False).mean()
    ema_fast_slow = safe_ratio_series(ema_fast - ema_slow, ema_slow)
    output["ema_fast_slow"] = ema_fast_slow
    output["rsi_1h"] = compute_rsi(close, config.normalized_rsi_length)

    atr_14 = compute_atr(
        pd.to_numeric(candles["high"], errors="coerce"),
        pd.to_numeric(candles["low"], errors="coerce"),
        close,
        length=14,
    )
    slope_12 = compute_linear_regression_slope(close, 12)
    slope_24 = compute_linear_regression_slope(close, 24)
    output["linear_regression_slope_atr_1h_12"] = safe_ratio_series(slope_12, atr_14)
    output["linear_regression_slope_atr_1h_24"] = safe_ratio_series(slope_24, atr_14)

    signed_step = pd.Series(np.sign(close.diff()), index=close.index)
    output["trend_persistence_score_12"] = signed_step.rolling(12).mean()
    output["trend_persistence_score_24"] = signed_step.rolling(24).mean()
    output["trend_efficiency_24h"] = compute_trend_efficiency(close, 24)
    output["slope_acceleration_1h_12_24"] = safe_ratio_series(slope_12 - slope_24, atr_14)
    output["ema_slope_acceleration_1h"] = ema_fast_slow - ema_fast_slow.shift(3)

    return output[["timestamp_ms", *MOMENTUM_FEATURE_COLUMNS]]
