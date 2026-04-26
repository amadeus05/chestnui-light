"""Реализации FeatureProvider для precomputed / realtime (legacy-семантика timestamps)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.application.backtest import backtest_data as ld
from src.contracts.feature_provider import FeatureProvider, PredictionProvider


class PrecomputedFeatureProvider(FeatureProvider):
    """Фичи из БД: строка на decision_ts (как в precomputed-ветке бэктеста)."""

    def __init__(
        self,
        *,
        all_features_by_symbol: dict[str, pd.DataFrame],
        feature_names: list[str],
        symbol_categories: list[str] | None,
        clip_bounds: dict[str, Any],
    ) -> None:
        self._raw = all_features_by_symbol
        self._feature_names = list(feature_names)
        self._symbol_categories = symbol_categories
        self._prepared = {
            sym: ld.prepare_precomputed_feature_store(
                df,
                self._feature_names,
                symbol_categories=symbol_categories,
                clip_bounds=clip_bounds,
            )
            for sym, df in all_features_by_symbol.items()
            if not df.empty
        }

    @property
    def prepared_stores(self) -> dict[str, pd.DataFrame]:
        return self._prepared

    def get_feature_row(self, symbol: str, timestamp: pd.Timestamp) -> pd.DataFrame | None:
        need = self._feature_names + ["barrier_stop_pct", "barrier_take_pct"]
        return ld.get_feature_row_precomputed(self._raw.get(symbol, pd.DataFrame()), timestamp, need)

    def batch_model_matrix(self, symbols: list[str], timestamp: pd.Timestamp) -> tuple[list[str], pd.DataFrame]:
        return ld.get_feature_batch_precomputed(self._prepared, symbols, timestamp)


class RealtimeFeatureProvider(FeatureProvider):
    """Пересчёт фич через FeaturePipeline на exec_ts (как в realtime-ветке бэктеста)."""

    def __init__(
        self,
        *,
        all_raw: dict,
        feature_names: list[str],
        symbol_categories: list[str] | None,
    ) -> None:
        self._all_raw = all_raw
        self._feature_names = list(feature_names)
        self._symbol_categories = symbol_categories

    def get_feature_row(self, symbol: str, timestamp: pd.Timestamp) -> pd.DataFrame | None:
        return ld.build_feature_row_at_time(
            self._all_raw,
            symbol,
            timestamp,
            self._feature_names,
            symbol_categories=self._symbol_categories,
        )


class DataFramePredictionProvider(PredictionProvider):
    """Walk-forward: словарь (timestamp, symbol) -> (p_short, p_long)."""

    def __init__(self, lookup: dict) -> None:
        self._lookup = lookup

    def get_proba(self, symbol: str, timestamp: pd.Timestamp) -> tuple[float, float] | None:
        key = (timestamp, symbol)
        return self._lookup.get(key)
