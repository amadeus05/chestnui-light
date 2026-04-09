from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, compute_donchian_channels, compute_lwti, safe_ratio
from src.features.models.feature_context import FeatureContext


class LwtiFeatureBuilder(FeatureBuilderContract):
    block_name = "donchian_lwti"

    def __init__(self) -> None:
        self.donchian_length = int(getattr(cfg, "DONCHIAN_LENGTH", 96))
        self.lwti_period = int(getattr(cfg, "LWTI_PERIOD", 25))
        self.lwti_smoothing_period = int(getattr(cfg, "LWTI_SMOOTHING_PERIOD", 20))
        self.volume_ma_length = int(getattr(cfg, "VOLUME_MA_LENGTH", 30))
        self.htf_sr_lookback = int(getattr(cfg, "HTF_SR_LOOKBACK", 24))
        self.htf_sr_buffer_atr = float(getattr(cfg, "HTF_SR_BUFFER_ATR", 0.5))

    def provides(self) -> set[str]:
        return {
            "donchian_width_pct_96",
            "donchian_mid_distance_atr_96",
            "donchian_close_position_96",
            "donchian_upper_break_atr_96",
            "donchian_lower_break_atr_96",
            "donchian_upper_touch_96",
            "donchian_lower_touch_96",
            "lwti_25_20",
            "lwti_centered_25_20",
            "lwti_slope_3",
            "lwti_long_bias_25_20",
            "volume_ratio_30",
            "volume_green_1",
            "volume_red_1",
            "volume_green_above_ma_30",
            "volume_red_above_ma_30",
            "htf_resistance_distance_atr_24",
            "htf_support_distance_atr_24",
            "htf_near_resistance_24",
            "htf_near_support_24",
            "breakout_long_score",
            "breakout_short_score",
            "breakout_long_signal",
            "breakout_short_signal",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        frame = context.frame
        open_ = frame["open"]
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        volume = frame["volume"]
        cache = context.indicator_cache

        atr_14 = cache.get_or_create(
            "donchian_lwti:atr_14",
            lambda: compute_atr(high, low, close, length=14),
        )
        donchian_upper, donchian_mid, donchian_lower = cache.get_or_create(
            f"donchian_lwti:donchian:{self.donchian_length}",
            lambda: compute_donchian_channels(high, low, length=self.donchian_length, shift=1),
        )
        donchian_range = (donchian_upper - donchian_lower).replace(0, np.nan)
        donchian_valid = donchian_upper.notna() & donchian_lower.notna() & atr_14.notna()
        upper_touch = self._float_flag(high >= donchian_upper, donchian_valid)
        lower_touch = self._float_flag(low <= donchian_lower, donchian_valid)
        upper_break_atr = safe_ratio((high - donchian_upper).clip(lower=0.0), atr_14)
        lower_break_atr = safe_ratio((donchian_lower - low).clip(lower=0.0), atr_14)

        lwti = cache.get_or_create(
            f"donchian_lwti:lwti:{self.lwti_period}:{self.lwti_smoothing_period}",
            lambda: compute_lwti(
                close=close,
                high=high,
                low=low,
                period=self.lwti_period,
                smoothing_period=self.lwti_smoothing_period,
            ),
        )
        lwti_centered = lwti - 50.0
        lwti_valid = lwti.notna()
        lwti_long_bias = self._float_flag(lwti > 50.0, lwti_valid)
        lwti_short_bias = self._float_flag(lwti < 50.0, lwti_valid)

        volume_ma = cache.get_or_create(
            f"donchian_lwti:volume_ma:{self.volume_ma_length}",
            lambda: volume.rolling(self.volume_ma_length, min_periods=self.volume_ma_length).mean(),
        )
        volume_ratio = safe_ratio(volume, volume_ma)
        volume_green = (close > open_).astype(float)
        volume_red = (close < open_).astype(float)
        volume_ma_valid = volume_ma.notna()
        volume_green_above_ma = self._float_flag((volume > volume_ma) & (close > open_), volume_ma_valid)
        volume_red_above_ma = self._float_flag((volume > volume_ma) & (close < open_), volume_ma_valid)

        htf_levels = self._align_htf_levels(context)
        htf_resistance_distance_atr = safe_ratio(htf_levels["htf_resistance"] - close, htf_levels["htf_atr_14"])
        htf_support_distance_atr = safe_ratio(close - htf_levels["htf_support"], htf_levels["htf_atr_14"])
        htf_resistance_valid = htf_resistance_distance_atr.notna() & htf_levels["htf_resistance"].notna()
        htf_support_valid = htf_support_distance_atr.notna() & htf_levels["htf_support"].notna()
        htf_near_resistance = self._float_flag(
            (htf_levels["htf_resistance"] >= close)
            & (htf_resistance_distance_atr <= self.htf_sr_buffer_atr),
            htf_resistance_valid,
        )
        htf_near_support = self._float_flag(
            (htf_levels["htf_support"] <= close)
            & (htf_support_distance_atr <= self.htf_sr_buffer_atr),
            htf_support_valid,
        )

        breakout_long_parts = pd.concat(
            [upper_touch, lwti_long_bias, volume_green_above_ma, 1.0 - htf_near_resistance],
            axis=1,
        )
        breakout_short_parts = pd.concat(
            [lower_touch, lwti_short_bias, volume_red_above_ma, 1.0 - htf_near_support],
            axis=1,
        )
        breakout_long_valid = breakout_long_parts.notna().all(axis=1)
        breakout_short_valid = breakout_short_parts.notna().all(axis=1)
        breakout_long_score = breakout_long_parts.sum(axis=1, min_count=4)
        breakout_short_score = breakout_short_parts.sum(axis=1, min_count=4)
        breakout_long_signal = self._float_flag(
            (upper_touch > 0.0)
            & (lwti_long_bias > 0.0)
            & (volume_green_above_ma > 0.0)
            & (htf_near_resistance == 0.0),
            breakout_long_valid,
        )
        breakout_short_signal = self._float_flag(
            (lower_touch > 0.0)
            & (lwti_short_bias > 0.0)
            & (volume_red_above_ma > 0.0)
            & (htf_near_support == 0.0),
            breakout_short_valid,
        )

        feature_map = {
            "donchian_width_pct_96": safe_ratio(donchian_range, close).abs(),
            "donchian_mid_distance_atr_96": safe_ratio(close - donchian_mid, atr_14),
            "donchian_close_position_96": safe_ratio(close - donchian_lower, donchian_range),
            "donchian_upper_break_atr_96": upper_break_atr,
            "donchian_lower_break_atr_96": lower_break_atr,
            "donchian_upper_touch_96": upper_touch,
            "donchian_lower_touch_96": lower_touch,
            "lwti_25_20": lwti,
            "lwti_centered_25_20": lwti_centered,
            "lwti_slope_3": lwti - lwti.shift(3),
            "lwti_long_bias_25_20": lwti_long_bias,
            "volume_ratio_30": volume_ratio,
            "volume_green_1": volume_green,
            "volume_red_1": volume_red,
            "volume_green_above_ma_30": volume_green_above_ma,
            "volume_red_above_ma_30": volume_red_above_ma,
            "htf_resistance_distance_atr_24": htf_resistance_distance_atr,
            "htf_support_distance_atr_24": htf_support_distance_atr,
            "htf_near_resistance_24": htf_near_resistance,
            "htf_near_support_24": htf_near_support,
            "breakout_long_score": breakout_long_score,
            "breakout_short_score": breakout_short_score,
            "breakout_long_signal": breakout_long_signal,
            "breakout_short_signal": breakout_short_signal,
        }

        for feature_name in sorted(active):
            output[feature_name] = feature_map[feature_name]
        return output

    def _align_htf_levels(self, context: FeatureContext) -> pd.DataFrame:
        base_frame = context.frame[["timestamp", "close"]].copy().sort_values("timestamp").reset_index(drop=True)
        empty_output = base_frame[["timestamp"]].copy()
        empty_output["htf_resistance"] = np.nan
        empty_output["htf_support"] = np.nan
        empty_output["htf_atr_14"] = np.nan

        htf_map = context.htf_feature_map or {}
        htf_frame = htf_map.get(context.symbol)
        if htf_frame is None or htf_frame.empty:
            return empty_output

        htf_frame = htf_frame.sort_values("timestamp").reset_index(drop=True)
        htf_high = htf_frame["high"]
        htf_low = htf_frame["low"]
        htf_close = htf_frame["close"]

        htf_features = htf_frame[["timestamp"]].copy()
        htf_features["htf_resistance"] = htf_high.rolling(
            self.htf_sr_lookback,
            min_periods=self.htf_sr_lookback,
        ).max().shift(1)
        htf_features["htf_support"] = htf_low.rolling(
            self.htf_sr_lookback,
            min_periods=self.htf_sr_lookback,
        ).min().shift(1)
        htf_features["htf_atr_14"] = compute_atr(htf_high, htf_low, htf_close, length=14)

        base_merge = base_frame[["timestamp"]].copy()
        base_merge["_merge_ts"] = base_merge["timestamp"] + self._timeframe_to_timedelta(
            str(getattr(cfg, "TIMEFRAME", "1h"))
        )

        htf_merge = htf_features.copy()
        htf_merge["_merge_ts"] = htf_merge["timestamp"] + self._timeframe_to_timedelta(
            str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        )

        merged = pd.merge_asof(
            base_merge.sort_values("_merge_ts").reset_index(drop=True),
            htf_merge[["_merge_ts", "htf_resistance", "htf_support", "htf_atr_14"]]
            .sort_values("_merge_ts")
            .reset_index(drop=True),
            on="_merge_ts",
            direction="backward",
        )
        return merged.drop(columns=["_merge_ts"], errors="ignore")

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
        amount = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        if unit == "m":
            return pd.to_timedelta(amount, unit="m")
        if unit == "h":
            return pd.to_timedelta(amount, unit="h")
        if unit == "d":
            return pd.to_timedelta(amount, unit="d")
        if unit == "w":
            return pd.to_timedelta(amount * 7, unit="d")
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    @staticmethod
    def _float_flag(condition: pd.Series, valid_mask: pd.Series) -> pd.Series:
        return pd.Series(
            np.where(valid_mask, condition.astype(float), np.nan),
            index=condition.index,
            dtype="float64",
        )
