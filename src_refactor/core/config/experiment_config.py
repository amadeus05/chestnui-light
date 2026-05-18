from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src_refactor.core.types import ModelSpec, SplitMode


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelSpec
    split_mode: SplitMode = "monthly_expanding"
    symbols: tuple[str, ...] = ()
    n_splits: int = 5
    train_months: int = 6
    test_months: int = 1
    purge_gap: int = 0
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    target_column: str = "Target"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        return self.model.model_id
