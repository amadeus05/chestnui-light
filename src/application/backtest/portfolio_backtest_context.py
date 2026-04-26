"""Контекст портфельного бэктеста для TradingEngine (фичи, модель, сырые данные)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.application.backtest.backtest_config import BacktestRunConfig


@dataclass
class PortfolioBacktestContext:
    """Данные для фазы входов на каждом шаге (precomputed / realtime / walk-forward)."""

    run_config: BacktestRunConfig
    feature_names: list[str]
    event_filter_config: dict
    symbol_categories: list[str] | None
    clip_bounds: dict[str, Any]
    all_raw: dict
    all_features: dict[str, pd.DataFrame]
    all_features_prepared: dict[str, pd.DataFrame]
    all_main_index: dict[str, dict]
    model: Any | None = None
    prediction_lookup: dict | None = None

    @property
    def using_external_predictions(self) -> bool:
        return bool(self.prediction_lookup)
