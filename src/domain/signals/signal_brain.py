"""SignalBrain — ML модуль для генерации торговых сигналов.

Поддерживает два источника предсказаний:
- feature_provider: строит фичи на лету, модель вызывается в цикле
- prediction_provider: walk-forward предсказания без модели (приоритет)

Если ни один не передан — сигналы не генерируются.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from src.contracts.feature_provider import FeatureProvider, PredictionProvider


class SignalPrediction(NamedTuple):
    """Расширенный результат предсказания — для backtest-ранжирования кандидатов."""

    signal: int  # -1 short, 1 long
    direction_prob: float  # вероятность победившего направления
    signal_gap: float  # |p_long - p_short| — сила сигнала
    stop_pct: float
    take_pct: float
    p_long: float
    p_short: float


def _resolve_event_filter(event_filter_meta: dict | None) -> dict:
    """Создает конфиг event filter из метаданных модели."""
    if not event_filter_meta:
        return {"enabled": False}
    try:
        from signal_filter import resolve_event_filter_config

        return resolve_event_filter_config(event_filter_meta)
    except ImportError:
        return {"enabled": False}


class SignalBrain:
    """
    Интерфейс к ML модели для генерации торговых сигналов.

    Загружает LightGBM модель и предсказывает:
    - signal: -1 (short), 0 (нет сигнала), 1 (long)
    - probability: уверенность модели
    - stop_pct / take_pct: уровни барьеров

    Два режима предсказаний (priority = prediction_provider):
    1. prediction_provider — готовые вероятности (walk-forward OOS)
    2. feature_provider + model — вычисление на лету
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        features_meta_path: str | Path | None = None,
        threshold: float = 0.5,
        min_signal_gap: float = 0.0,
        allow_longs: bool = True,
        allow_shorts: bool = True,
        feature_provider: FeatureProvider | None = None,
        prediction_provider: PredictionProvider | None = None,
    ) -> None:
        self._model = None
        self._feature_names: list[str] = []
        self._threshold = threshold
        self._min_signal_gap = min_signal_gap
        self._allow_longs = allow_longs
        self._allow_shorts = allow_shorts
        self._clip_bounds: dict = {}
        self._symbol_categories: list[str] | None = None
        self._event_filter_config: dict = {"enabled": False}

        self._feature_provider = feature_provider
        self._prediction_provider = prediction_provider

        if model_path and features_meta_path:
            self.load_model(model_path, features_meta_path)

    # ── загрузка ──────────────────────────────────────────────────────────────

    def load_model(
        self, model_path: str | Path, features_meta_path: str | Path
    ) -> None:
        """Загружает модель и метаданные."""
        import joblib

        model_path = Path(model_path)
        features_meta_path = Path(features_meta_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not features_meta_path.exists():
            raise FileNotFoundError(f"Features meta not found: {features_meta_path}")

        self._model = joblib.load(model_path)

        with open(features_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._feature_names = meta.get("feature_columns", [])

        clip_meta = meta.get("feature_clip", {})
        self._clip_bounds = clip_meta.get("bounds", {})

        if "symbols" in meta:
            self._symbol_categories = meta["symbols"]

        self._event_filter_config = _resolve_event_filter(meta.get("event_filter"))

        print(f"[SignalBrain] Loaded model from {model_path}")
        print(f"  Features: {len(self._feature_names)}")
        if self._event_filter_config.get("enabled"):
            print("  Event filter: enabled")

    def set_feature_provider(self, provider: FeatureProvider) -> None:
        self._feature_provider = provider

    def set_prediction_provider(self, provider: PredictionProvider) -> None:
        self._prediction_provider = provider

    def apply_runtime_config(
        self,
        *,
        model: Any | None = None,
        feature_names: list[str],
        clip_bounds: dict | None = None,
        symbol_categories: list[str] | None = None,
        event_filter_config: dict | None = None,
    ) -> None:
        """
        Настройка без load_model() — для portfolio_backtest_runner, где модель и meta уже загружены.
        """
        if model is not None:
            self._model = model
        self._feature_names = list(feature_names)
        self._clip_bounds = dict(clip_bounds or {})
        self._symbol_categories = symbol_categories
        if event_filter_config is not None:
            self._event_filter_config = event_filter_config

    # ── основные методы ───────────────────────────────────────────────────────

    def predict(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> tuple[int, float, float, float] | None:
        """
        Предсказывает сигнал для символа на заданный момент.

        Returns:
            (signal, probability, stop_pct, take_pct) или None
            signal: -1 (short), 0 (нет сигнала), 1 (long)
        """
        full = self.predict_full(symbol, timestamp)
        if full is None:
            return None
        return full.signal, full.direction_prob, full.stop_pct, full.take_pct

    def predict_full(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> SignalPrediction | None:
        """
        Расширенная версия predict() — возвращает signal_gap для ранжирования.

        Используется в portfolio_backtest_runner для вычисления entry score.
        TradingEngine использует более простой predict().

        Returns:
            SignalPrediction или None (нет сигнала)
        """
        # Путь 1: готовые вероятности из walk-forward (приоритет)
        if self._prediction_provider is not None:
            return self._predict_full_from_provider(symbol, timestamp)

        # Путь 2: модель + feature_provider
        if self._model is not None and self._feature_provider is not None:
            return self._predict_full_from_model(symbol, timestamp)

        return None

    # ── внутренние методы ─────────────────────────────────────────────────────

    def _predict_full_from_provider(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> SignalPrediction | None:
        """Walk-forward: вероятности из lookup, барьеры + event filter из feature_provider."""
        assert self._prediction_provider is not None

        proba = self._prediction_provider.get_proba(symbol, timestamp)
        if proba is None:
            return None
        p_short, p_long = float(proba[0]), float(proba[1])

        if self._feature_provider is None:
            return None
        feature_row = self._feature_provider.get_feature_row(symbol, timestamp)
        if feature_row is None or feature_row.empty:
            return None
        if not self._passes_event_filter(feature_row):
            return None
        stop_pct, take_pct = self._extract_barrier_pcts(feature_row)
        if stop_pct is None or take_pct is None:
            return None

        return self._build_full_result(p_long, p_short, stop_pct, take_pct)

    def _predict_full_from_model(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> SignalPrediction | None:
        """Модель + feature_provider."""
        assert self._model is not None
        assert self._feature_provider is not None

        feature_row = self._feature_provider.get_feature_row(symbol, timestamp)
        if feature_row is None or feature_row.empty:
            return None

        if not self._passes_event_filter(feature_row):
            return None

        features = self._prepare_feature_matrix(feature_row)
        if features is None:
            return None

        stop_pct, take_pct = self._extract_barrier_pcts(feature_row)
        if stop_pct is None or take_pct is None:
            return None

        try:
            proba = self._model.predict_proba(features)[0]
        except Exception as e:
            print(f"[SignalBrain] predict_proba failed for {symbol}: {e}")
            return None

        p_short = float(proba[0])
        p_long = float(proba[1])
        return self._build_full_result(p_long, p_short, stop_pct, take_pct)

    def _build_full_result(
        self, p_long: float, p_short: float, stop_pct: float, take_pct: float
    ) -> SignalPrediction | None:
        signal, direction_prob, signal_gap = self._resolve_signal(p_long, p_short)
        if signal == 0:
            return None
        return SignalPrediction(
            signal=signal,
            direction_prob=direction_prob,
            signal_gap=signal_gap,
            stop_pct=stop_pct,
            take_pct=take_pct,
            p_long=p_long,
            p_short=p_short,
        )

    def _prepare_feature_matrix(self, feature_row: pd.DataFrame) -> pd.DataFrame | None:
        """Отбирает нужные колонки, применяет clip bounds."""
        if not self._feature_names:
            return None
        missing = [f for f in self._feature_names if f not in feature_row.columns]
        if missing:
            return None
        features = feature_row[self._feature_names].copy()
        if "symbol" in features.columns and self._symbol_categories is not None:
            features["symbol"] = pd.Categorical(
                features["symbol"], categories=self._symbol_categories
            )
        if self._clip_bounds:
            for col, bounds in self._clip_bounds.items():
                if col in features.columns:
                    features[col] = features[col].clip(
                        lower=bounds["lower"], upper=bounds["upper"]
                    )
        if features.isna().any(axis=None):
            return None
        return features

    def _extract_barrier_pcts(
        self, feature_row: pd.DataFrame
    ) -> tuple[float | None, float | None]:
        if (
            "barrier_stop_pct" not in feature_row.columns
            or "barrier_take_pct" not in feature_row.columns
        ):
            return None, None
        stop = float(feature_row["barrier_stop_pct"].iloc[0])
        take = float(feature_row["barrier_take_pct"].iloc[0])
        if not np.isfinite(stop) or not np.isfinite(take):
            return None, None
        return stop, take

    def _passes_event_filter(self, feature_row: pd.DataFrame) -> bool:
        if not self._event_filter_config.get("enabled", False):
            return True
        try:
            from signal_filter import build_candidate_event_mask

            mask = build_candidate_event_mask(feature_row, self._event_filter_config)
            return bool(mask.iloc[0]) if not mask.empty else False
        except ImportError:
            return True

    def _resolve_signal(
        self, p_long: float, p_short: float
    ) -> tuple[int, float, float]:
        signal_gap = abs(p_long - p_short)
        if (
            p_long >= self._threshold
            and (p_long - p_short) >= self._min_signal_gap
            and self._allow_longs
        ):
            return (1, p_long, signal_gap)
        if (
            p_short >= self._threshold
            and (p_short - p_long) >= self._min_signal_gap
            and self._allow_shorts
        ):
            return (-1, p_short, signal_gap)
        return (0, max(p_long, p_short), signal_gap)

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()
