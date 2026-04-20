from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class BtcRelativeFeatureBuilder(FeatureBuilderContract):
    block_name = "btc_relative"

    def provides(self) -> set[str]:
        return {
            "relative_strength_vs_btc_24h",
            "beta_to_btc_24h",
            "residual_return_24h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.base_feature_map is None:
            raise ValueError("BTC-relative feature block requires base_feature_map.")

        cache_key = "btc_reference_frame"
        btc_reference = context.shared_cache.get(cache_key)
        if btc_reference is None:
            btc_df = context.base_feature_map.get("BTC/USDT")
            if btc_df is None or btc_df.empty:
                btc_reference = pd.DataFrame(columns=["timestamp", "btc_close", "btc_return_1h_24"])
            else:
                btc_reference = btc_df[["timestamp", "close"]].copy().rename(
                    columns={
                        "close": "btc_close",
                    }
                )
                if "return_1h_24" in btc_df.columns:
                    btc_reference["btc_return_1h_24"] = btc_df["return_1h_24"].values
                else:
                    btc_reference["btc_return_1h_24"] = np.log(
                        btc_reference["btc_close"] / btc_reference["btc_close"].shift(24)
                    )
            context.shared_cache[cache_key] = btc_reference

        merged = context.frame[["timestamp", "close"]].merge(
            btc_reference,
            on="timestamp",
            how="left",
        )
        if "return_1h_24" in context.frame.columns:
            merged["asset_return_1h_24"] = context.frame["return_1h_24"].values
        else:
            merged["asset_return_1h_24"] = np.log(merged["close"] / merged["close"].shift(24))
        if merged["btc_close"].isna().all():
            for feature_name in sorted(active):
                output[feature_name] = np.nan
            return output

        asset_return_1h = np.log(merged["close"] / merged["close"].shift(1))
        btc_return_1h = np.log(merged["btc_close"] / merged["btc_close"].shift(1))
        btc_var_24h = btc_return_1h.rolling(24).var().replace(0, np.nan)
        beta_24h = asset_return_1h.rolling(24).cov(btc_return_1h) / btc_var_24h

        if "relative_strength_vs_btc_24h" in active:
            output["relative_strength_vs_btc_24h"] = merged["asset_return_1h_24"] - merged["btc_return_1h_24"]
        if "beta_to_btc_24h" in active:
            output["beta_to_btc_24h"] = beta_24h
        if "residual_return_24h" in active:
            output["residual_return_24h"] = merged["asset_return_1h_24"] - (beta_24h * merged["btc_return_1h_24"])

        if context.symbol == "BTC/USDT":
            if "relative_strength_vs_btc_24h" in active:
                output["relative_strength_vs_btc_24h"] = 0.0
            if "beta_to_btc_24h" in active:
                output["beta_to_btc_24h"] = 1.0
            if "residual_return_24h" in active:
                output["residual_return_24h"] = 0.0

        return output
