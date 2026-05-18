from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from src_refactor.core.types import MarketDataBatch


class MarketBatchStream(Protocol):
    def stream(self) -> Iterable[MarketDataBatch]:
        raise NotImplementedError
