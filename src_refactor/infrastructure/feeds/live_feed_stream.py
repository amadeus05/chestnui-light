from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd

from src_refactor.core.contracts import MarketBatchStream
from src_refactor.core.contracts.stream_market_feed import LiveMarketDataFeed
from src_refactor.core.types import Candle, MarketDataBatch, MarketDataEvent, MarketDataSubscription


@dataclass(frozen=True, slots=True)
class _StreamEnd:
    error: BaseException | None = None


@dataclass(slots=True)
class LiveFeedMarketStream(MarketBatchStream):
    feed: LiveMarketDataFeed
    subscriptions: list[MarketDataSubscription]
    _queue: queue.Queue[MarketDataBatch | _StreamEnd] = field(init=False, default_factory=queue.Queue)

    def stream(self) -> Iterator[MarketDataBatch]:
        producer = threading.Thread(target=self._run_feed, daemon=True)
        producer.start()
        while True:
            item = self._queue.get()
            if isinstance(item, _StreamEnd):
                producer.join()
                if item.error is not None:
                    raise item.error
                return
            yield item

    def _run_feed(self) -> None:
        error: BaseException | None = None
        try:
            asyncio.run(self._produce())
        except BaseException as exc:
            error = exc
        finally:
            self._queue.put(_StreamEnd(error=error))

    async def _produce(self) -> None:
        batcher = _TimestampBatcher(self.subscriptions)

        def on_event(event: MarketDataEvent) -> None:
            batch = batcher.add(event.candle)
            if batch is not None:
                self._queue.put(batch)

        try:
            await self.feed.subscribe(self.subscriptions)
            await self.feed.run(on_event)
        finally:
            for batch in batcher.flush():
                self._queue.put(batch)
            await self.feed.close()


@dataclass(slots=True)
class _TimestampBatcher:
    subscriptions: list[MarketDataSubscription]
    pending: dict[pd.Timestamp, dict[str, Candle]] = field(default_factory=dict)

    @property
    def expected_symbols(self) -> set[str]:
        return {subscription.symbol for subscription in self.subscriptions}

    def add(self, candle: Candle) -> MarketDataBatch | None:
        timestamp = pd.to_datetime(candle.timestamp)
        symbol_map = self.pending.setdefault(timestamp, {})
        symbol_map[candle.symbol] = candle
        expected_symbols = self.expected_symbols
        if expected_symbols and not expected_symbols.issubset(symbol_map):
            return None
        candles = tuple(symbol_map[symbol] for symbol in sorted(symbol_map))
        del self.pending[timestamp]
        return MarketDataBatch(timestamp=timestamp, candles=candles)

    def flush(self) -> tuple[MarketDataBatch, ...]:
        batches = tuple(
            MarketDataBatch(timestamp=timestamp, candles=tuple(symbol_map.values()))
            for timestamp, symbol_map in sorted(self.pending.items(), key=lambda item: item[0])
            if symbol_map
        )
        self.pending.clear()
        return batches
