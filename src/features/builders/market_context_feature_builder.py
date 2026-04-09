from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, safe_ratio
from src.features.models.feature_context import FeatureContext


class MarketContextFeatureBuilder(FeatureBuilderContract):
    block_name = "market_context"

    btc_symbol = "BTC/USDT"

    def __init__(self) -> None:
        self.return_24h_bars = self._bars_for_hours(24)
        self.return_4h_bars = self._bars_for_hours(4)
        self.return_3d_bars = self._bars_for_days(3)
        self.corr_window = int(getattr(cfg, "MARKET_CONTEXT_CORR_WINDOW", 96))
        self.atr_length = int(getattr(cfg, "MARKET_CONTEXT_ATR_LENGTH", 14))

    def provides(self) -> set[str]:
        return {
            "btc_return_24h",
            "relative_strength_vs_btc_24h",
            "relative_return_vs_btc_4h",
            "relative_return_vs_btc_3d",
            "corr_to_btc_96",
            "beta_to_btc_96",
            "atr_pct_ratio_to_btc",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        base_feature_map = context.base_feature_map or {}
        btc_frame = base_feature_map.get(self.btc_symbol)
        if btc_frame is None or btc_frame.empty:
            raise ValueError("Market context features require BTC/USDT candles in the base feature map.")

        asset_frame = context.frame.sort_values("timestamp").reset_index(drop=True)
        asset_close = asset_frame["close"]
        asset_high = asset_frame["high"]
        asset_low = asset_frame["low"]
        asset_log_return_1 = np.log(asset_close / asset_close.shift(1))
        asset_return_24h = np.log(asset_close / asset_close.shift(self.return_24h_bars))
        asset_return_4h = np.log(asset_close / asset_close.shift(self.return_4h_bars))
        asset_return_3d = np.log(asset_close / asset_close.shift(self.return_3d_bars))
        asset_atr_pct = safe_ratio(compute_atr(asset_high, asset_low, asset_close, self.atr_length), asset_close).abs()

        btc_reference = btc_frame[["timestamp", "high", "low", "close"]].copy().sort_values("timestamp").reset_index(drop=True)
        btc_close = btc_reference["close"]
        btc_log_return_1 = np.log(btc_close / btc_close.shift(1))
        btc_reference["btc_close"] = btc_close
        btc_reference["btc_return_24h"] = np.log(btc_close / btc_close.shift(self.return_24h_bars))
        btc_reference["btc_return_4h"] = np.log(btc_close / btc_close.shift(self.return_4h_bars))
        btc_reference["btc_return_3d"] = np.log(btc_close / btc_close.shift(self.return_3d_bars))
        btc_reference["btc_atr_pct"] = safe_ratio(
            compute_atr(btc_reference["high"], btc_reference["low"], btc_close, self.atr_length),
            btc_close,
        ).abs()
        merged = asset_frame[["timestamp"]].merge(
            btc_reference[
                ["timestamp", "btc_close", "btc_return_24h", "btc_return_4h", "btc_return_3d", "btc_atr_pct"]
            ],
            on="timestamp",
            how="left",
        )
        merged["btc_log_return_1"] = np.log(merged["btc_close"] / merged["btc_close"].shift(1))
        merged["btc_corr_96"] = asset_log_return_1.rolling(self.corr_window).corr(merged["btc_log_return_1"])
        btc_var_96 = merged["btc_log_return_1"].rolling(self.corr_window).var().replace(0, np.nan)
        merged["btc_beta_96"] = asset_log_return_1.rolling(self.corr_window).cov(merged["btc_log_return_1"]) / btc_var_96

        if "relative_strength_vs_btc_24h" in active:
            if context.symbol == self.btc_symbol:
                merged["relative_strength_vs_btc_24h"] = 0.0
            else:
                merged["relative_strength_vs_btc_24h"] = asset_return_24h.values - merged["btc_return_24h"].values
        if "relative_return_vs_btc_4h" in active:
            if context.symbol == self.btc_symbol:
                merged["relative_return_vs_btc_4h"] = 0.0
            else:
                merged["relative_return_vs_btc_4h"] = asset_return_4h.values - merged["btc_return_4h"].values
        if "relative_return_vs_btc_3d" in active:
            if context.symbol == self.btc_symbol:
                merged["relative_return_vs_btc_3d"] = 0.0
            else:
                merged["relative_return_vs_btc_3d"] = asset_return_3d.values - merged["btc_return_3d"].values
        if "corr_to_btc_96" in active:
            merged["corr_to_btc_96"] = 1.0 if context.symbol == self.btc_symbol else merged["btc_corr_96"].values
        if "beta_to_btc_96" in active:
            merged["beta_to_btc_96"] = 1.0 if context.symbol == self.btc_symbol else merged["btc_beta_96"].values
        if "atr_pct_ratio_to_btc" in active:
            if context.symbol == self.btc_symbol:
                merged["atr_pct_ratio_to_btc"] = 1.0
            else:
                merged["atr_pct_ratio_to_btc"] = safe_ratio(asset_atr_pct, merged["btc_atr_pct"]).values

        merged = merged.drop(columns=["btc_close", "btc_log_return_1"], errors="ignore")
        if "btc_return_24h" not in active:
            merged = merged.drop(columns=["btc_return_24h"], errors="ignore")
        if "relative_return_vs_btc_4h" not in active:
            merged = merged.drop(columns=["btc_return_4h"], errors="ignore")
        if "relative_return_vs_btc_3d" not in active:
            merged = merged.drop(columns=["btc_return_3d"], errors="ignore")
        if "corr_to_btc_96" not in active:
            merged = merged.drop(columns=["btc_corr_96"], errors="ignore")
        if "beta_to_btc_96" not in active:
            merged = merged.drop(columns=["btc_beta_96"], errors="ignore")
        if "atr_pct_ratio_to_btc" not in active:
            merged = merged.drop(columns=["btc_atr_pct"], errors="ignore")
        return merged

    @staticmethod
    def _bars_for_hours(hours: int) -> int:
        minutes = MarketContextFeatureBuilder._timeframe_to_minutes(str(getattr(cfg, "TIMEFRAME", "5m")))
        return max(1, int(round((hours * 60) / minutes)))

    @staticmethod
    def _bars_for_days(days: int) -> int:
        minutes = MarketContextFeatureBuilder._timeframe_to_minutes(str(getattr(cfg, "TIMEFRAME", "5m")))
        return max(1, int(round((days * 24 * 60) / minutes)))

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        amount = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        if unit == "m":
            return amount
        if unit == "h":
            return amount * 60
        if unit == "d":
            return amount * 24 * 60
        if unit == "w":
            return amount * 7 * 24 * 60
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
