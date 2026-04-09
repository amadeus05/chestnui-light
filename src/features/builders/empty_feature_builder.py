from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class EmptyFeatureBuilder(FeatureBuilderContract):
    block_name = "empty"

    def provides(self) -> set[str]:
        return set()

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        raise NotImplementedError("EmptyFeatureBuilder does not implement any features.")
