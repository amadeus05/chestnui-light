"""Абстракция источника OHLCV данных."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.contracts.clock import Clock


@dataclass(frozen=True)
class Bar:
    """Один OHLCV бар."""

    symbol: str
    timestamp: pd.Timestamp  # время закрытия
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: pd.Timestamp  # время открытия

    @property
    def hl2(self) -> float:
        """Среднее high-low."""
        return (self.high + self.low) / 2.0

    @property
    def hlc3(self) -> float:
        """Среднее high-low-close."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def ohlc4(self) -> float:
        """Среднее open-high-low-close."""
        return (self.open + self.high + self.low + self.close) / 4.0


class DataFeed(ABC):
    """Источник OHLCV данных для TradingEngine.

    Реализации:
    - ReplayDataFeed: исторические данные для бэктеста
    - LiveDataFeed: websocket данные для paper/live
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    @property
    def clock(self) -> Clock:
        return self._clock

    @abstractmethod
    def initialize(self, symbols: list[str], timeframe: str) -> None:
        """Подготовка: загрузка истории или подписка на потоки."""
        raise NotImplementedError

    @abstractmethod
    def next_bar_batch(self) -> dict[str, Bar] | None:
        """
        Возвращает словарь {symbol: Bar} для следующего временного шага.

        Для бэктеста: следующий timestamp из истории.
        Для live: ожидание закрытия бара по websocket.

        Returns:
            dict[str, Bar] | None: None когда данные закончились (бэктест)
                или остановка запрошена.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mark_prices(self) -> dict[str, float]:
        """Текущие mark prices (close последнего бара) для всех символов."""
        raise NotImplementedError

    def now(self) -> pd.Timestamp:
        """Текущее время от Clock."""
        return self._clock.now()
