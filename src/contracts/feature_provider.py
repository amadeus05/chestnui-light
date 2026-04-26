"""Абстракция поставщика фич для SignalBrain / бэктеста (расширение по мере миграции)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureProvider(ABC):
    """Одна строка фич для (symbol, timestamp) в семантике legacy."""

    @abstractmethod
    def get_feature_row(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """DataFrame из одной строки или None."""


class PredictionProvider(ABC):
    """Вероятности (p_short, p_long) для walk-forward без модели в цикле."""

    @abstractmethod
    def get_proba(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> tuple[float, float] | None:
        ...
