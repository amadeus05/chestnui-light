from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class TimeContextFeatureBuilder(FeatureBuilderContract):
    block_name = "time_context"

    def provides(self) -> set[str]:
        return {
            "hour_sin_1h",
            "hour_cos_1h",
            "is_weekend_1h",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        hour = frame["timestamp"].dt.hour
        if "hour_sin_1h" in active:
            output["hour_sin_1h"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_cos_1h" in active:
            output["hour_cos_1h"] = np.cos(2.0 * np.pi * hour / 24.0)
        if "is_weekend_1h" in active:
            output["is_weekend_1h"] = (frame["timestamp"].dt.dayofweek >= 5).astype(float)
        return output
