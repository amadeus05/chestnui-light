from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from src_refactor.core.types import MarketDataEvent, MarketDataSubscription


class LiveMarketDataFeed(ABC):
    @abstractmethod
    async def subscribe(self, subscriptions: list[MarketDataSubscription]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def run(self, on_event: Callable[[MarketDataEvent], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
