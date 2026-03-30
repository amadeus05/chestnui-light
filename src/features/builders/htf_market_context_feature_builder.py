from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class HtfMarketContextFeatureBuilder(FeatureBuilderContract):
    block_name = "market_context"

    def provides(self) -> set[str]:
        return {
            "market_breadth_pos_return_4h_3",
            "market_dispersion_return_4h_3",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.htf_feature_map is None:
            raise ValueError("HTF market context block requires htf_feature_map.")
        if "return_4h_3" not in context.frame.columns:
            raise ValueError("HTF market context requires 'return_4h_3'.")

        cache_key = "market_context:htf:return_4h_3"
        context_df = context.shared_cache.get(cache_key)
        if context_df is None:
            frames = []
            for source_df in context.htf_feature_map.values():
                if "return_4h_3" not in source_df.columns:
                    continue
                frame = source_df[["timestamp", "return_4h_3"]].dropna(subset=["return_4h_3"]).copy()
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                context_df = pd.DataFrame(
                    columns=[
                        "timestamp",
                        "market_breadth_pos_return_4h_3",
                        "market_dispersion_return_4h_3",
                    ]
                )
            else:
                context_source = pd.concat(frames, ignore_index=True)
                grouped = context_source.groupby("timestamp")["return_4h_3"]
                context_df = pd.DataFrame({"timestamp": grouped.size().index})
                context_df["market_breadth_pos_return_4h_3"] = grouped.apply(lambda series: float((series > 0).mean())).values
                context_df["market_dispersion_return_4h_3"] = grouped.apply(
                    lambda series: float(np.nanstd(series.to_numpy(dtype=float), ddof=0))
                ).values
            context.shared_cache[cache_key] = context_df

        return output.merge(context_df, on="timestamp", how="left")
