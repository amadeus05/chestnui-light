from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

SplitMode: TypeAlias = Literal["tscv", "monthly_expanding", "monthly_rolling"]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_gap: int = 0

    def train_mask(self, timestamps: pd.Series) -> pd.Series:
        return timestamps.between(self.train_start, self.train_end, inclusive="both")

    def test_mask(self, timestamps: pd.Series) -> pd.Series:
        return timestamps.between(self.test_start, self.test_end, inclusive="both")
