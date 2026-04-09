from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class InteractionFeatureBuilder(FeatureBuilderContract):
    block_name = "interaction"

    def provides(self) -> set[str]:
        return set()

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        raise NotImplementedError("InteractionFeatureBuilder does not implement any features.")
