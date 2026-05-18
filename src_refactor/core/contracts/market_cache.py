from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src_refactor.core.types import Candle, Symbol


class MarketCache(ABC):
    @abstractmethod
    def append_candle(self, candle: Candle) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_candles(self, symbol: Symbol, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        raise NotImplementedError
