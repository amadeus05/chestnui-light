"""Абстракция времени: wall-clock (лайв) vs replay (бэктест по барам)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timezone

import pandas as pd


class Clock(ABC):
    """Текущий логический момент для движка (решения, снимки equity, логирование)."""

    @abstractmethod
    def now(self) -> pd.Timestamp:
        raise NotImplementedError


class WallClock(Clock):
    """Реальное UTC-время (paper / live)."""

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=timezone.utc)


class ReplayClock(Clock):
    """Симуляция: время задаётся снаружи на каждом шаге (закрытие текущего бара в бэктесте)."""

    __slots__ = ("_current",)

    def __init__(self, start: pd.Timestamp | str | None = None) -> None:
        self._current: pd.Timestamp | None = None if start is None else pd.Timestamp(start)

    def set(self, ts: pd.Timestamp | str) -> None:
        self._current = pd.Timestamp(ts)

    def now(self) -> pd.Timestamp:
        if self._current is None:
            raise RuntimeError("ReplayClock: вызовите set(ts) до now()")
        return self._current
