from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class TrendFeatureBuilder(FeatureBuilderContract):
    block_name = "trend"

    def __init__(self, timeframe_label: str) -> None:
        self.timeframe_label = timeframe_label

    def provides(self) -> set[str]:
        return set()

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        raise NotImplementedError("TrendFeatureBuilder does not implement any features.")
