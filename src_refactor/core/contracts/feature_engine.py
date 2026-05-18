from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src_refactor.core.types import ModelInput, Symbol


class FeatureEngine(ABC):
    @abstractmethod
    def build(self, symbol: Symbol, timeframe: str) -> ModelInput | None:
        raise NotImplementedError
