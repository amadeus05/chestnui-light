from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class BetaTestFeatureBuilder(FeatureBuilderContract):
    block_name = "beta_test"

    BASE_RETURN_PERIODS = (1, 3, 6, 12, 24, 48, 72)
    HTF_RETURN_PERIODS = (1, 3, 6)

    def __init__(self, timeframe: str) -> None:
        self.timeframe = timeframe

    def provides(self) -> set[str]:
        if self.timeframe == "1h":
            return {f"return_1h_{period}" for period in self.BASE_RETURN_PERIODS}
        if self.timeframe == "4h":
            return {f"return_4h_{period}" for period in self.HTF_RETURN_PERIODS}
        return set()

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = pd.to_numeric(frame["close"], errors="coerce")
        periods = self.BASE_RETURN_PERIODS if self.timeframe == "1h" else self.HTF_RETURN_PERIODS
        prefix = f"return_{self.timeframe}_"
        for period in periods:
            feature_name = f"{prefix}{period}"
            if feature_name in active:
                output[feature_name] = np.log(close / close.shift(period))

        if self.timeframe == "4h":
            feature_columns = [column for column in output.columns if column != "timestamp"]
            output[feature_columns] = output[feature_columns].shift(1)

        return output
