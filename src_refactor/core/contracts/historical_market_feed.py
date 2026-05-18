from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src_refactor.core.types import MarketDataEvent


class HistoricalMarketFeed(ABC):
    @abstractmethod
    def events(self) -> Iterator[MarketDataEvent]:
        raise NotImplementedError