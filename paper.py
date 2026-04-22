"""
Пейпер-трейдинг: синхронизация по BTC (основной ТФ + 1m), сделки в SQLite.

- BTC 1h: определяем момент «закрытся час на бирже» и запускаем анализ всех SYMBOLS.
- BTC 1m: тот же «биржевой» тайминг между часовыми прогонами (последняя закрытая минута).
- Выходы TP/SL внутри часа: по 1m свечам **торгуемого символа** (иначе нельзя
  восстановить порядок касаний для альтов). BTC 1m здесь не используется.

Таблица execution_trades: execution_type IN ('paper','live') — для live потом тот же журнал.

Запуск:
  python paper.py           # по умолчанию: цикл до Ctrl+C (то же, что paper.py daemon)
  python paper.py tick      # один проход (cron / ручная проверка)
  python paper.py status    # открытые + баланс
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

import config as cfg
import etl
from paper_api import start_background_api
from lstm.model import LSTMClassifier
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.application.entry_candidate_resolution import resolve_and_build_entry_candidate
from src.bootstrap.container import build_execution_stack_from_config
from src.execution import (
    ExecutionEngine,
    ExecutionPortfolio,
    ExecutionRunner,
    PortfolioState,
    Position,
    RuntimePosition,
)
from src.exchanges.bybit.bybit_kline_stream import BybitKlineStream, KlineClosedEvent
from src.features import MasterFeatureBuilder
from src.persistence.repositories.base_trades_repository import BaseTradesRepository
from src.persistence.repositories.trades_repository_factory import create_trades_repository_from_config

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("paper")

STATE_LAST_HOUR = "paper_last_processed_main_bar_open_ms_btc"
EXEC_TYPE = "paper"
MARKET_CACHE_OVERLAP_BARS = 2
FUNDING_CACHE_OVERLAP_POINTS = 2
_MARKET_DATA_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_EXECUTION_STACK = build_execution_stack_from_config(cfg)
EXECUTION_ENGINE = _EXECUTION_STACK.engine
EXECUTION_PORTFOLIO = _EXECUTION_STACK.portfolio
EXECUTION_RUNNER = _EXECUTION_STACK.runner


@dataclass
class ModelBundle:
    model_name: str
    model_type: str
    feature_names: list[str]
    event_filter_config: dict
    clip_bounds: dict
    symbol_categories: list[str] | None
    model: object
    sequence_length: int = 1
    standardizer: dict | None = None


@dataclass
class PaperRuntime:
    repo: BaseTradesRepository
    model_bundle: ModelBundle
    exchange: object
    execution_runner: ExecutionRunner
    execution_engine: ExecutionEngine
    clock_sym: str
    main_tf: str
    main_tf_ms: int
    tf_1m_ms: int
    signal_engine: object | None = None
    candidate_builder: object | None = None
    predictor: object | None = None


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _build_portfolio_state(repo: BaseTradesRepository) -> PortfolioState:
    initial = float(getattr(cfg, "PAPER_INITIAL_BALANCE", getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)))
    leverage = float(getattr(cfg, "LEVERAGE", 1))
    balance = initial + repo.sum_closed_pnl_quote(EXEC_TYPE)
    used_margin = repo.sum_open_margin_quote(EXEC_TYPE, leverage)
    return PortfolioState(
        initial_balance=initial,
        balance=balance,
        used_margin=used_margin,
    )


def _runtime_position_from_trade(trade: dict) -> RuntimePosition:
    entry_ts_ms = int(trade.get("entry_ts_ms") or 0)
    required_margin = float(trade["entry_notional"]) / max(float(getattr(cfg, "LEVERAGE", 1.0)), 1e-12)
    return RuntimePosition(
        symbol=str(trade["symbol"]),
        trade_number=int(trade.get("id", 0)),
        direction=int(trade["direction"]),
        entry_price=float(trade["entry_price"]),
        notional=float(trade["entry_notional"]),
        required_margin=required_margin,
        stop_pct=float(trade["stop_pct"]),
        take_pct=float(trade["take_pct"]),
        opened_at=datetime.fromtimestamp(entry_ts_ms / 1000, tz=timezone.utc),
    )


def _series_open_ms(ts: pd.Series) -> np.ndarray:
    # Pandas can store datetime64 with microsecond precision on some platforms.
    # Convert explicitly to datetime64[ms] so downstream exchange requests always use epoch milliseconds.
    return ts.astype("datetime64[ms]").astype("int64").to_numpy(dtype=np.int64)


def _cache_key(kind: str, symbol: str, timeframe: str) -> tuple[str, str, str]:
    return kind, symbol, timeframe


def _clone_cached_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.copy(deep=False)


def _merge_time_series_frames(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    keep_last_rows: int | None = None,
) -> pd.DataFrame:
    if current.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = current.copy()
    else:
        merged = pd.concat([current, incoming], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    if keep_last_rows is not None and len(merged) > keep_last_rows:
        merged = merged.tail(keep_last_rows)
    return merged.reset_index(drop=True)


def _get_cached_frame(kind: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    cached = _MARKET_DATA_CACHE.get(_cache_key(kind, symbol, timeframe))
    if cached is None:
        return None
    return _clone_cached_frame(cached)


def _set_cached_frame(kind: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    _MARKET_DATA_CACHE[_cache_key(kind, symbol, timeframe)] = df.copy()


def _mask_fully_closed(df: pd.DataFrame, tf_ms: int, now_ms: int) -> pd.Series:
    o = _series_open_ms(df["timestamp"])
    return pd.Series(o + tf_ms <= now_ms, index=df.index)


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    rows = [
        (k.open_time, k.open, k.high, k.low, k.close, k.volume)
        for k in sorted(klines, key=lambda x: x.open_time)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _fetch_ohlcv_bars(exchange, symbol: str, timeframe: str, min_bars: int) -> pd.DataFrame:
    tf_ms = exchange.get_timeframe_ms(timeframe)
    end_ts = _utc_now_ms()
    start_ts = end_ts - int(min_bars * tf_ms)
    cached = _get_cached_frame("ohlcv", symbol, timeframe)

    if cached is None or cached.empty:
        klines = exchange.fetch_klines(symbol, timeframe, start_ts, end_ts)
        fresh = _klines_to_dataframe(klines)
        result = fresh.tail(min_bars).reset_index(drop=True)
        _set_cached_frame("ohlcv", symbol, timeframe, result)
        return result

    cached_open_ms = _series_open_ms(cached["timestamp"])
    cached_start_ts = int(cached_open_ms[0])
    cached_end_ts = int(cached_open_ms[-1])

    merged = cached

    if start_ts < cached_start_ts:
        backfill_end = min(cached_start_ts - tf_ms, end_ts)
        if start_ts <= backfill_end:
            older = _klines_to_dataframe(exchange.fetch_klines(symbol, timeframe, start_ts, backfill_end))
            merged = _merge_time_series_frames(older, merged)

    update_start = max(start_ts, cached_end_ts - tf_ms * MARKET_CACHE_OVERLAP_BARS)
    if update_start <= end_ts:
        newer = _klines_to_dataframe(exchange.fetch_klines(symbol, timeframe, update_start, end_ts))
        merged = _merge_time_series_frames(merged, newer)

    result = merged.tail(min_bars).reset_index(drop=True)
    _set_cached_frame("ohlcv", symbol, timeframe, result)
    return result


def _fetch_funding_context(exchange, symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    timeframe = "8h"
    cached = _get_cached_frame("funding", symbol, timeframe)
    interval_ms = int(getattr(exchange, "get_funding_interval_ms")(symbol))

    if cached is None or cached.empty:
        points = exchange.fetch_funding_rates(symbol, start_ts, end_ts)
        fresh = _funding_points_to_dataframe(points)
        _set_cached_frame("funding", symbol, timeframe, fresh)
        return fresh

    cached_open_ms = _series_open_ms(cached["timestamp"])
    cached_start_ts = int(cached_open_ms[0])
    cached_end_ts = int(cached_open_ms[-1])
    merged = cached

    if start_ts < cached_start_ts:
        backfill_end = min(cached_start_ts - interval_ms, end_ts)
        if start_ts <= backfill_end:
            older = _funding_points_to_dataframe(exchange.fetch_funding_rates(symbol, start_ts, backfill_end))
            merged = _merge_time_series_frames(older, merged)

    update_start = max(start_ts, cached_end_ts - interval_ms * FUNDING_CACHE_OVERLAP_POINTS)
    if update_start <= end_ts:
        newer = _funding_points_to_dataframe(exchange.fetch_funding_rates(symbol, update_start, end_ts))
        merged = _merge_time_series_frames(merged, newer)

    _set_cached_frame("funding", symbol, timeframe, merged)
    return merged


def _fetch_premium_context(exchange, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    tf_ms = exchange.get_timeframe_ms(timeframe)
    cached = _get_cached_frame("premium", symbol, timeframe)

    if cached is None or cached.empty:
        premium_klines = exchange.fetch_premium_index_klines(symbol, timeframe, start_ts, end_ts)
        premium_df = _klines_to_dataframe(premium_klines)
        if premium_df.empty:
            return pd.DataFrame(columns=["timestamp", "premium_index_close"])
        premium_df = premium_df[["timestamp", "close"]].rename(columns={"close": "premium_index_close"})
        _set_cached_frame("premium", symbol, timeframe, premium_df)
        return premium_df

    cached_open_ms = _series_open_ms(cached["timestamp"])
    cached_start_ts = int(cached_open_ms[0])
    cached_end_ts = int(cached_open_ms[-1])
    merged = cached

    if start_ts < cached_start_ts:
        backfill_end = min(cached_start_ts - tf_ms, end_ts)
        if start_ts <= backfill_end:
            older_raw = _klines_to_dataframe(exchange.fetch_premium_index_klines(symbol, timeframe, start_ts, backfill_end))
            older = (
                older_raw[["timestamp", "close"]].rename(columns={"close": "premium_index_close"})
                if not older_raw.empty
                else pd.DataFrame(columns=["timestamp", "premium_index_close"])
            )
            merged = _merge_time_series_frames(older, merged)

    update_start = max(start_ts, cached_end_ts - tf_ms * MARKET_CACHE_OVERLAP_BARS)
    if update_start <= end_ts:
        newer_raw = _klines_to_dataframe(exchange.fetch_premium_index_klines(symbol, timeframe, update_start, end_ts))
        newer = (
            newer_raw[["timestamp", "close"]].rename(columns={"close": "premium_index_close"})
            if not newer_raw.empty
            else pd.DataFrame(columns=["timestamp", "premium_index_close"])
        )
        merged = _merge_time_series_frames(merged, newer)

    _set_cached_frame("premium", symbol, timeframe, merged)
    return merged


def _fetch_open_interest_context(exchange, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    tf_ms = exchange.get_timeframe_ms(timeframe)
    cached = _get_cached_frame("open_interest", symbol, timeframe)

    if cached is None or cached.empty:
        points = exchange.fetch_open_interest(symbol, timeframe, start_ts, end_ts)
        fresh = _open_interest_points_to_dataframe(points)
        _set_cached_frame("open_interest", symbol, timeframe, fresh)
        return fresh

    cached_open_ms = _series_open_ms(cached["timestamp"])
    cached_start_ts = int(cached_open_ms[0])
    cached_end_ts = int(cached_open_ms[-1])
    merged = cached

    if start_ts < cached_start_ts:
        backfill_end = min(cached_start_ts - tf_ms, end_ts)
        if start_ts <= backfill_end:
            older = _open_interest_points_to_dataframe(exchange.fetch_open_interest(symbol, timeframe, start_ts, backfill_end))
            merged = _merge_time_series_frames(older, merged)

    update_start = max(start_ts, cached_end_ts - tf_ms * MARKET_CACHE_OVERLAP_BARS)
    if update_start <= end_ts:
        newer = _open_interest_points_to_dataframe(exchange.fetch_open_interest(symbol, timeframe, update_start, end_ts))
        merged = _merge_time_series_frames(merged, newer)

    _set_cached_frame("open_interest", symbol, timeframe, merged)
    return merged


def _funding_points_to_dataframe(points: list) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    rows = [
        (point.funding_time, point.funding_rate)
        for point in sorted(points, key=lambda x: x.funding_time)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _open_interest_points_to_dataframe(points: list) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    rows = [
        (point.timestamp, point.open_interest)
        for point in sorted(points, key=lambda x: x.timestamp)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _fetch_symbol_feature_inputs(
    exchange,
    symbol: str,
    main_tf: str,
    htf: str,
    main_bars: int,
    htf_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_df = _fetch_ohlcv_bars(exchange, symbol, main_tf, main_bars)
    hdf = _fetch_ohlcv_bars(exchange, symbol, htf, htf_bars)
    if main_df.empty or hdf.empty:
        return main_df, hdf

    start_ts = int(_series_open_ms(main_df["timestamp"])[0])
    end_ts = _utc_now_ms()

    try:
        funding_df = _fetch_funding_context(exchange, symbol, start_ts, end_ts)
    except Exception:
        logger.exception("%s: funding context load failed", symbol)
        funding_df = pd.DataFrame(columns=["timestamp", "funding_rate"])

    try:
        premium_df = _fetch_premium_context(exchange, symbol, main_tf, start_ts, end_ts)
    except Exception:
        logger.exception("%s: premium index context load failed", symbol)
        premium_df = pd.DataFrame(columns=["timestamp", "premium_index_close"])

    try:
        open_interest_df = _fetch_open_interest_context(exchange, symbol, main_tf, start_ts, end_ts)
    except NotImplementedError:
        logger.warning("%s: open interest context is not supported by the current exchange adapter", symbol)
        open_interest_df = pd.DataFrame(columns=["timestamp", "open_interest"])
    except Exception:
        logger.exception("%s: open interest context load failed", symbol)
        open_interest_df = pd.DataFrame(columns=["timestamp", "open_interest"])

    main_with_context = etl.attach_funding_context(main_df, funding_df)
    main_with_context = etl.attach_premium_index_context(main_with_context, premium_df)
    main_with_context = etl.attach_open_interest_context(main_with_context, open_interest_df)
    return main_with_context, hdf


def _standardize_sequence(sequence: np.ndarray, standardizer_payload: dict | None) -> np.ndarray:
    if not standardizer_payload:
        return sequence.astype(np.float32)
    mean = np.asarray(standardizer_payload.get("mean", []), dtype=np.float32)
    std = np.asarray(standardizer_payload.get("std", []), dtype=np.float32)
    if mean.size == 0 or std.size == 0:
        return sequence.astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((sequence.astype(np.float32) - mean) / std).astype(np.float32)


def _infer_model_type(model_name: str) -> str:
    configured = str(getattr(cfg, "PAPER_MODEL_TYPE", "auto")).strip().lower()
    if configured in {"lightgbm", "lstm"}:
        return configured
    models_dir = Path(getattr(cfg, "MODELS_DIR", "models"))
    if (models_dir / f"{model_name}.joblib").is_file():
        return "lightgbm"
    if (models_dir / f"{model_name}.pt").is_file():
        return "lstm"
    return "lightgbm"


def _load_model_bundle() -> ModelBundle:
    models_dir = Path(getattr(cfg, "MODELS_DIR", "models"))
    model_name = str(
        getattr(
            cfg,
            "PAPER_MODEL_NAME",
            getattr(cfg, "PAPER_LSTM_MODEL_NAME", "lstm_target_production"),
        )
    )
    model_type = _infer_model_type(model_name)

    if model_type == "lightgbm":
        model_path = models_dir / f"{model_name}.joblib"
        meta_path = models_dir / f"{model_name}_features.json"
        if not model_path.is_file() or not meta_path.is_file():
            logger.error("Нужны %s и %s (train.py)", model_path, meta_path)
            sys.exit(1)

        with open(meta_path, encoding="utf-8") as f:
            features_meta = json.load(f)
        feature_names = list(features_meta["feature_columns"])
        trained_symbols = list(features_meta.get("symbols", []))
        event_filter_meta = features_meta.get("event_filter")
        if event_filter_meta is None:
            logger.error("В JSON нет event_filter — переобучите train.py")
            sys.exit(1)
        event_filter_config = resolve_event_filter_config(event_filter_meta)
        clip_bounds = (features_meta.get("feature_clip") or {}).get("bounds") or {}
        use_symbol = "symbol" in feature_names
        scan_symbols = list(dict.fromkeys(str(s) for s in getattr(cfg, "SYMBOLS", [])))
        symbol_categories = list(dict.fromkeys(trained_symbols + scan_symbols)) if use_symbol else None
        model = joblib.load(model_path)
        return ModelBundle(
            model_name=model_name,
            model_type=model_type,
            feature_names=feature_names,
            event_filter_config=event_filter_config,
            clip_bounds=clip_bounds,
            symbol_categories=symbol_categories,
            model=model,
        )

    if torch is None:
        logger.error("Для LSTM execution нужен torch: pip install torch")
        sys.exit(1)

    model_path = models_dir / f"{model_name}.pt"
    meta_path = models_dir / f"{model_name}_features.json"
    if not model_path.is_file() or not meta_path.is_file():
        logger.error("Нужны %s и %s (lstm/train_lstm_production.py)", model_path, meta_path)
        sys.exit(1)

    payload = torch.load(model_path, map_location="cpu")
    with open(meta_path, encoding="utf-8") as f:
        features_meta = json.load(f)

    feature_names = list(payload.get("feature_columns") or features_meta.get("feature_columns") or [])
    if not feature_names:
        logger.error("У LSTM-модели пустой список feature_columns: %s", model_path)
        sys.exit(1)

    sequence_length = int(payload.get("sequence_length") or features_meta.get("sequence_length") or 48)
    standardizer = payload.get("standardizer") or {}
    model_args = payload.get("model_args") or {}
    event_filter_meta = features_meta.get("event_filter") or {"enabled": False}
    event_filter_config = resolve_event_filter_config(event_filter_meta)
    clip_bounds = (features_meta.get("feature_clip") or {}).get("bounds") or {}
    trained_symbols = list(features_meta.get("symbols", []))
    use_symbol = "symbol" in feature_names
    scan_symbols = list(dict.fromkeys(str(s) for s in getattr(cfg, "SYMBOLS", [])))
    symbol_categories = list(dict.fromkeys(trained_symbols + scan_symbols)) if use_symbol else None

    model = LSTMClassifier(
        input_size=int(model_args.get("input_size", len(feature_names))),
        hidden_size=int(model_args.get("hidden_size", 64)),
        num_layers=int(model_args.get("num_layers", 1)),
        dropout=float(model_args.get("dropout", 0.2)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ModelBundle(
        model_name=model_name,
        model_type=model_type,
        feature_names=feature_names,
        event_filter_config=event_filter_config,
        clip_bounds=clip_bounds,
        symbol_categories=symbol_categories,
        model=model,
        sequence_length=sequence_length,
        standardizer=standardizer,
    )


def _model_bundle_from_injected(
    model: object,
    features_meta: dict,
    *,
    model_name: str | None = None,
) -> ModelBundle:
    """Сборка ModelBundle из runtime-injected model + metadata (без чтения с диска)."""
    resolved_name = model_name or str(
        getattr(
            cfg,
            "PAPER_MODEL_NAME",
            getattr(cfg, "PAPER_LSTM_MODEL_NAME", "lstm_target_production"),
        )
    )
    meta_type = str(features_meta.get("model_type") or "").strip().lower()
    if meta_type in {"lstm", "lightgbm"}:
        model_type = meta_type
    elif isinstance(model, LSTMClassifier):
        model_type = "lstm"
    elif hasattr(model, "predict_proba"):
        model_type = "lightgbm"
    else:
        model_type = _infer_model_type(resolved_name)

    feature_names = list(features_meta["feature_columns"])
    trained_symbols = list(features_meta.get("symbols", []))
    event_filter_meta = features_meta.get("event_filter")
    if event_filter_meta is None:
        logger.error("В injected features_meta нет event_filter")
        sys.exit(1)
    event_filter_config = resolve_event_filter_config(event_filter_meta)
    clip_bounds = (features_meta.get("feature_clip") or {}).get("bounds") or {}
    use_symbol = "symbol" in feature_names
    scan_symbols = list(dict.fromkeys(str(s) for s in getattr(cfg, "SYMBOLS", [])))
    symbol_categories = list(dict.fromkeys(trained_symbols + scan_symbols)) if use_symbol else None

    if model_type == "lstm":
        if torch is None:
            logger.error("Для LSTM execution нужен torch: pip install torch")
            sys.exit(1)
        sequence_length = int(features_meta.get("sequence_length") or 48)
        standardizer = features_meta.get("standardizer") or {}
        return ModelBundle(
            model_name=resolved_name,
            model_type="lstm",
            feature_names=feature_names,
            event_filter_config=event_filter_config,
            clip_bounds=clip_bounds,
            symbol_categories=symbol_categories,
            model=model,
            sequence_length=sequence_length,
            standardizer=standardizer if standardizer else None,
        )

    return ModelBundle(
        model_name=resolved_name,
        model_type="lightgbm",
        feature_names=feature_names,
        event_filter_config=event_filter_config,
        clip_bounds=clip_bounds,
        symbol_categories=symbol_categories,
        model=model,
    )


def _predict_symbol_proba(
    model_bundle: ModelBundle,
    row: pd.DataFrame,
    feat_df: pd.DataFrame,
    predictor=None,
) -> tuple[float, float] | None:
    if model_bundle.model_type == "lightgbm":
        x = _normalize_features_for_model(row, model_bundle.feature_names, model_bundle.symbol_categories)
        x = _apply_feature_clip_bounds(x, model_bundle.clip_bounds)
        active_predictor = predictor or model_bundle.model
        proba = active_predictor.predict_proba(x)[0]
        return float(proba[0]), float(proba[1])

    if torch is None:
        return None

    seq_len = int(model_bundle.sequence_length)
    if len(feat_df) < seq_len:
        return None

    feature_frame = feat_df.copy()
    if "symbol" in model_bundle.feature_names and "symbol" not in feature_frame.columns:
        feature_frame["symbol"] = row["symbol"].iloc[0]
    missing = [f for f in model_bundle.feature_names if f not in feature_frame.columns]
    if missing:
        return None
    seq_df = feature_frame.tail(seq_len).copy()
    if "symbol" in model_bundle.feature_names:
        seq_df["symbol"] = pd.Categorical(
            seq_df["symbol"].astype(str),
            categories=model_bundle.symbol_categories,
        ).codes.astype(np.float32)
    sequence = seq_df[model_bundle.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    sequence = _standardize_sequence(sequence, model_bundle.standardizer)
    x = torch.from_numpy(sequence[None, :, :])
    with torch.no_grad():
        probs = torch.softmax(model_bundle.model(x), dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1])


def _telegram_enabled() -> bool:
    return bool(getattr(cfg, "TELEGRAM_NOTIFICATIONS_ENABLED", False))


def _send_telegram_message(message: str) -> None:
    if not _telegram_enabled():
        return
    bot_token = getattr(cfg, "TELEGRAM_BOT_TOKEN", None)
    chat_id = getattr(cfg, "TELEGRAM_CHAT_ID", None)
    if not bot_token or not chat_id:
        logger.warning("Telegram notifications enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not configured")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("Telegram notification send failed")


def _format_trade_open_message(
    *,
    trade_id: int,
    symbol: str,
    direction: int,
    entry_price: float,
    stop_pct: float,
    take_pct: float,
    p_long: float,
    p_short: float,
    notional: float,
    model_name: str,
) -> str:
    direction_label = "LONG" if direction == 1 else "SHORT"
    stop_price = entry_price * (1.0 - stop_pct) if direction == 1 else entry_price * (1.0 + stop_pct)
    take_price = entry_price * (1.0 + take_pct) if direction == 1 else entry_price * (1.0 - take_pct)
    rr = take_pct / stop_pct if stop_pct > 0 else 0.0
    return (
        "<b>OPEN</b>\n"
        f"ID: <code>{trade_id}</code>\n"
        f"Model: <code>{model_name}</code>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Side: <b>{direction_label}</b>\n"
        f"Entry: <code>{entry_price:.6f}</code>\n"
        f"Stop: <code>{stop_price:.6f}</code> ({stop_pct * 100:.2f}%)\n"
        f"Take: <code>{take_price:.6f}</code> ({take_pct * 100:.2f}%)\n"
        f"RR: <code>{rr:.2f}</code>\n"
        f"Notional: <code>{notional:.2f}</code>\n"
        f"p_long: <code>{p_long:.3f}</code>\n"
        f"p_short: <code>{p_short:.3f}</code>"
    )


def _format_trade_close_message(
    *,
    trade_id: int,
    symbol: str,
    direction: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    pnl_pct: float,
    pnl_quote: float,
) -> str:
    direction_label = "LONG" if direction == 1 else "SHORT"
    return (
        "<b>CLOSE</b>\n"
        f"ID: <code>{trade_id}</code>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Side: <b>{direction_label}</b>\n"
        f"Entry: <code>{entry_price:.6f}</code>\n"
        f"Exit: <code>{exit_price:.6f}</code>\n"
        f"Reason: <b>{exit_reason}</b>\n"
        f"PnL %: <code>{pnl_pct:.2f}</code>\n"
        f"PnL $: <code>{pnl_quote:.2f}</code>"
    )


def _last_closed_bar_open_ms(df: pd.DataFrame, tf_ms: int, now_ms: int) -> int | None:
    if df is None or df.empty:
        return None
    m = _mask_fully_closed(df, tf_ms, now_ms)
    sub = df.loc[m]
    if sub.empty:
        return None
    o = _series_open_ms(sub["timestamp"])
    return int(o[-1])


def _normalize_features_for_model(
    latest_row: pd.DataFrame,
    feature_names: list,
    symbol_categories: list | None,
) -> pd.DataFrame:
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def _apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict) -> pd.DataFrame:
    if not clip_bounds:
        return frame
    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def _get_barrier_pcts(feature_row: pd.DataFrame) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None
    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None
    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct


def _is_candidate_event(feature_row: pd.DataFrame, event_filter_config: dict) -> bool:
    if feature_row is None or feature_row.empty:
        return False
    mask = build_candidate_event_mask(feature_row, event_filter_config)
    return bool(mask.iloc[0]) if not mask.empty else False


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _clock_symbol() -> str:
    return str(getattr(cfg, "PAPER_CLOCK_SYMBOL", "BTC/USDT"))


def _fmt_utc_ms(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "-"
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log_trade_open(
    *,
    trade_id: int,
    symbol: str,
    direction: int,
    entry_price: float,
    notional: float,
    stop_pct: float,
    take_pct: float,
    p_long: float,
    p_short: float,
    signal_gap: float,
    score: float,
    signal_bar_open_ms: int,
    entry_ts_ms: int,
    model_name: str,
) -> None:
    logger.info(
        "OPEN | id=%s | symbol=%s | side=%s | entry=%.8g | notional=%.2f | stop_pct=%.4f | "
        "take_pct=%.4f | p_long=%.3f | p_short=%.3f | gap=%.4f | score=%.4f | "
        "signal_bar=%s | entry_ts=%s | model=%s",
        trade_id,
        symbol,
        "LONG" if direction == 1 else "SHORT",
        entry_price,
        notional,
        stop_pct,
        take_pct,
        p_long,
        p_short,
        signal_gap,
        score,
        _fmt_utc_ms(signal_bar_open_ms),
        _fmt_utc_ms(entry_ts_ms),
        model_name,
    )


def _log_trade_close(
    *,
    trade_id: int,
    symbol: str,
    direction: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    entry_notional: float,
    stop_pct: float,
    take_pct: float,
    pnl_pct: float,
    pnl_quote: float,
    fees_quote: float,
    entry_ts_ms: int | None,
    exit_ts_ms: int,
    trigger_candle_open_ms: int | None = None,
    trigger_candle_ohlc: tuple[float, float, float] | None = None,
) -> None:
    trigger_suffix = ""
    if trigger_candle_open_ms is not None and trigger_candle_ohlc is not None:
        open_price, high_price, low_price = trigger_candle_ohlc
        trigger_suffix = (
            f" | trigger_bar={_fmt_utc_ms(trigger_candle_open_ms)}"
            f" | 1m_ohlc(open/high/low)={open_price:.8g}/{high_price:.8g}/{low_price:.8g}"
        )

    logger.info(
        "CLOSE | id=%s | symbol=%s | side=%s | entry=%.8g | exit=%.8g | reason=%s | "
        "notional=%.2f | stop_pct=%.4f | take_pct=%.4f | pnl_pct=%.2f | pnl_quote=%.2f | "
        "fees=%.4f | entry_ts=%s | exit_ts=%s%s",
        trade_id,
        symbol,
        "LONG" if direction == 1 else "SHORT",
        entry_price,
        exit_price,
        exit_reason,
        entry_notional,
        stop_pct,
        take_pct,
        pnl_pct,
        pnl_quote,
        fees_quote,
        _fmt_utc_ms(entry_ts_ms),
        _fmt_utc_ms(exit_ts_ms),
        trigger_suffix,
    )


def _ws_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "")


def _ws_interval(main_tf: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    interval = mapping.get(main_tf)
    if interval is None:
        raise ValueError(f"Unsupported Bybit websocket timeframe: {main_tf}")
    return interval


def _build_runtime(
    exchange=None,
    execution_runner: ExecutionRunner | None = None,
    signal_engine=None,
    candidate_builder=None,
    execution_engine: ExecutionEngine | None = None,
    model=None,
    features_meta: dict | None = None,
    predictor=None,
    model_name: str | None = None,
) -> PaperRuntime:
    repo = create_trades_repository_from_config()
    repo.init_schema()
    if model is not None and features_meta is not None:
        model_bundle = _model_bundle_from_injected(model, features_meta, model_name=model_name)
    else:
        model_bundle = _load_model_bundle()
    exchange = exchange or etl.create_exchange_service()
    execution_runner = execution_runner or EXECUTION_RUNNER
    if execution_engine is None:
        execution_engine = execution_runner.engine
    clock_sym = str(exchange.normalize_symbol(_clock_symbol()))
    main_tf = str(getattr(cfg, "TIMEFRAME", "1h"))
    main_tf_ms = exchange.get_timeframe_ms(main_tf)
    tf_1m_ms = exchange.get_timeframe_ms("1m")
    return PaperRuntime(
        repo=repo,
        model_bundle=model_bundle,
        exchange=exchange,
        execution_runner=execution_runner,
        execution_engine=execution_engine,
        clock_sym=clock_sym,
        main_tf=main_tf,
        main_tf_ms=main_tf_ms,
        tf_1m_ms=tf_1m_ms,
        signal_engine=signal_engine,
        candidate_builder=candidate_builder,
        predictor=predictor,
    )


def _db_path() -> str:
    p = getattr(cfg, "EXECUTION_DB_PATH", None)
    if p:
        return str(p)
    return str(cfg.DB_PATH)


def _fetch_closed_1m_range(
    exchange,
    symbol: str,
    start_open_ms: int,
    end_open_ms_exclusive: int,
    tf_ms_1m: int,
) -> pd.DataFrame:
    """Свечи 1m с open_time в [start, end)."""
    if start_open_ms >= end_open_ms_exclusive:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    pad_ms = tf_ms_1m * 5
    klines = exchange.fetch_klines(symbol, "1m", start_open_ms - pad_ms, end_open_ms_exclusive + pad_ms)
    df = _klines_to_dataframe(klines)
    if df.empty:
        return df
    o = _series_open_ms(df["timestamp"])
    keep = (o >= start_open_ms) & (o < end_open_ms_exclusive)
    return df.loc[keep].reset_index(drop=True)


def _process_minute_exits_for_trade(
    exchange,
    trade,
    *,
    last_closed_1m_open_ms: int,
    now_ms: int,
    tf_1m_ms: int,
    execution_runner: ExecutionRunner,
) -> bool:
    """Проверка 1m баров символа позиции. Возвращает True, если сделка закрыта."""
    repo: BaseTradesRepository = trade["repo"]
    tid = trade["id"]
    start_from = trade["last_1m_scan_open_ms"]
    if start_from is None:
        start_from = int(trade["entry_ts_ms"])

    end_open_exclusive = last_closed_1m_open_ms + tf_1m_ms
    if end_open_exclusive <= start_from:
        return False

    df = _fetch_closed_1m_range(exchange, trade["symbol"], start_from, end_open_exclusive, tf_1m_ms)
    now_ms = _utc_now_ms()
    mclosed = _mask_fully_closed(df, tf_1m_ms, now_ms)
    df = df.loc[mclosed].reset_index(drop=True)
    if df.empty:
        return False

    entry_price = float(trade["entry_price"])
    direction = int(trade["direction"])
    stop_pct = float(trade["stop_pct"])
    take_pct = float(trade["take_pct"])

    next_scan_from = start_from
    for _, row in df.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        exit_price, reason = etl.resolve_trade_exit(
            direction, entry_price, o, h, l, stop_pct, take_pct
        )
        open_ms = int(_series_open_ms(pd.Series([row["timestamp"]]))[0])
        bar_close_ms = open_ms + tf_1m_ms
        next_scan_from = bar_close_ms
        if exit_price is not None and reason:
            portfolio = _build_portfolio_state(repo)
            runtime_position = _runtime_position_from_trade(trade)
            closed_trade = execution_runner.close_position_on_bar(
                portfolio_state=portfolio,
                positions={str(trade["symbol"]): runtime_position},
                symbol=str(trade["symbol"]),
                next_open=o,
                next_high=h,
                next_low=l,
                closed_at=datetime.fromtimestamp(bar_close_ms / 1000, tz=timezone.utc),
            )
            if closed_trade is None:
                continue
            repo.close_trade(
                tid,
                exit_ts_ms=bar_close_ms,
                exit_price=float(closed_trade.exit_price),
                exit_reason=str(closed_trade.reason),
                pnl_pct=float(closed_trade.pnl_pct * 100.0),
                pnl_quote=float(closed_trade.pnl_abs),
                fees_quote=float(closed_trade.commission),
            )
            _log_trade_close(
                trade_id=tid,
                symbol=str(trade["symbol"]),
                direction=direction,
                entry_price=entry_price,
                exit_price=float(closed_trade.exit_price),
                exit_reason=str(closed_trade.reason),
                entry_notional=float(trade["entry_notional"]),
                stop_pct=stop_pct,
                take_pct=take_pct,
                pnl_pct=float(closed_trade.pnl_pct * 100.0),
                pnl_quote=float(closed_trade.pnl_abs),
                fees_quote=float(closed_trade.commission),
                entry_ts_ms=int(trade.get("entry_ts_ms")) if trade.get("entry_ts_ms") is not None else None,
                exit_ts_ms=bar_close_ms,
                trigger_candle_open_ms=open_ms,
                trigger_candle_ohlc=(o, h, l),
            )
            if bool(getattr(cfg, "TELEGRAM_NOTIFY_CLOSE", True)):
                _send_telegram_message(
                    _format_trade_close_message(
                        trade_id=tid,
                        symbol=str(trade["symbol"]),
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=float(closed_trade.exit_price),
                        exit_reason=str(closed_trade.reason),
                        pnl_pct=float(closed_trade.pnl_pct * 100.0),
                        pnl_quote=float(closed_trade.pnl_abs),
                    )
                )
            return True

    repo.update_last_1m_scan(tid, next_scan_from)
    return False


def _close_trade_from_1m_candle(
    trade: dict,
    *,
    candle_open_ms: int,
    tf_1m_ms: int,
    open_price: float,
    high_price: float,
    low_price: float,
    execution_runner: ExecutionRunner,
) -> bool:
    repo: BaseTradesRepository = trade["repo"]
    tid = trade["id"]
    entry_price = float(trade["entry_price"])
    direction = int(trade["direction"])
    stop_pct = float(trade["stop_pct"])
    take_pct = float(trade["take_pct"])

    bar_close_ms = candle_open_ms + tf_1m_ms
    portfolio = _build_portfolio_state(repo)
    runtime_position = _runtime_position_from_trade(trade)
    closed_trade = execution_runner.close_position_on_bar(
        portfolio_state=portfolio,
        positions={str(trade["symbol"]): runtime_position},
        symbol=str(trade["symbol"]),
        next_open=open_price,
        next_high=high_price,
        next_low=low_price,
        closed_at=datetime.fromtimestamp(bar_close_ms / 1000, tz=timezone.utc),
    )
    if closed_trade is None:
        repo.update_last_1m_scan(tid, bar_close_ms)
        return False
    repo.close_trade(
        tid,
        exit_ts_ms=bar_close_ms,
        exit_price=float(closed_trade.exit_price),
        exit_reason=str(closed_trade.reason),
        pnl_pct=float(closed_trade.pnl_pct * 100.0),
        pnl_quote=float(closed_trade.pnl_abs),
        fees_quote=float(closed_trade.commission),
    )
    _log_trade_close(
        trade_id=tid,
        symbol=str(trade["symbol"]),
        direction=direction,
        entry_price=entry_price,
        exit_price=float(closed_trade.exit_price),
        exit_reason=str(closed_trade.reason),
        entry_notional=float(trade["entry_notional"]),
        stop_pct=stop_pct,
        take_pct=take_pct,
        pnl_pct=float(closed_trade.pnl_pct * 100.0),
        pnl_quote=float(closed_trade.pnl_abs),
        fees_quote=float(closed_trade.commission),
        entry_ts_ms=int(trade.get("entry_ts_ms")) if trade.get("entry_ts_ms") is not None else None,
        exit_ts_ms=bar_close_ms,
        trigger_candle_open_ms=candle_open_ms,
        trigger_candle_ohlc=(open_price, high_price, low_price),
    )
    if bool(getattr(cfg, "TELEGRAM_NOTIFY_CLOSE", True)):
        _send_telegram_message(
            _format_trade_close_message(
                trade_id=tid,
                symbol=str(trade["symbol"]),
                direction=direction,
                entry_price=entry_price,
                exit_price=float(closed_trade.exit_price),
                exit_reason=str(closed_trade.reason),
                pnl_pct=float(closed_trade.pnl_pct * 100.0),
                pnl_quote=float(closed_trade.pnl_abs),
            )
        )
    return True


def _handle_1m_kline_close_event(
    repo: BaseTradesRepository,
    exchange,
    event: KlineClosedEvent,
    *,
    tf_1m_ms: int,
    execution_runner: ExecutionRunner,
) -> None:
    ws_symbol = _ws_symbol(str(event.symbol).upper())
    open_trades = [trade for trade in _load_open_trades_full(repo) if _ws_symbol(trade["symbol"]) == ws_symbol]
    if not open_trades:
        return

    for trade in open_trades:
        start_from = trade["last_1m_scan_open_ms"]
        if start_from is None:
            start_from = int(trade["entry_ts_ms"])

        if start_from < event.start_ms:
            _process_minute_exits_for_trade(
                exchange,
                trade,
                last_closed_1m_open_ms=event.start_ms,
                now_ms=_utc_now_ms(),
                tf_1m_ms=tf_1m_ms,
                execution_runner=execution_runner,
            )
            continue

        if start_from > event.start_ms:
            continue

        _close_trade_from_1m_candle(
            trade,
            candle_open_ms=event.start_ms,
            tf_1m_ms=tf_1m_ms,
            open_price=event.open_price,
            high_price=event.high_price,
            low_price=event.low_price,
            execution_runner=execution_runner,
        )


def _sync_ws_topics(
    stream: BybitKlineStream,
    repo: BaseTradesRepository,
    *,
    clock_sym: str,
    main_tf: str,
) -> None:
    topics = {f"kline.{_ws_interval(main_tf)}.{_ws_symbol(clock_sym)}"}
    for trade in _load_open_trades_full(repo):
        topics.add(f"kline.1.{_ws_symbol(trade['symbol'])}")
    stream.sync_topics(topics)


def _run_minute_exit_catchup(
    repo: BaseTradesRepository,
    exchange,
    execution_runner: ExecutionRunner,
) -> int | None:
    tf_1m_ms = exchange.get_timeframe_ms("1m")
    clock_sym = str(exchange.normalize_symbol(_clock_symbol()))
    now_ms = _utc_now_ms()
    btc_1m = _fetch_ohlcv_bars(exchange, clock_sym, "1m", 240)
    last_1m_open = _last_closed_bar_open_ms(btc_1m, tf_1m_ms, now_ms)
    if last_1m_open is None:
        logger.warning("Не удалось определить закрытый 1m бар для catch-up exits.")
        return None

    for tr in _load_open_trades_full(repo):
        _process_minute_exits_for_trade(
            exchange,
            tr,
            last_closed_1m_open_ms=last_1m_open,
            now_ms=now_ms,
            tf_1m_ms=tf_1m_ms,
            execution_runner=execution_runner,
        )
    return last_1m_open


def _run_main_timeframe_cycle(
    *,
    repo: BaseTradesRepository,
    model_bundle: ModelBundle,
    exchange,
    clock_sym: str,
    main_tf: str,
    main_tf_ms: int,
    execution_runner: ExecutionRunner,
    execution_engine: ExecutionEngine,
    signal_engine=None,
    candidate_builder=None,
    predictor=None,
) -> int | None:
    now_ms = _utc_now_ms()
    btc_1h = _fetch_ohlcv_bars(exchange, clock_sym, main_tf, 48)
    last_hour_open = _last_closed_bar_open_ms(btc_1h, main_tf_ms, now_ms)
    if last_hour_open is None:
        logger.warning("Не удалось определить закрытый бар %s для %s.", main_tf, clock_sym)
        return None

    prev_hour = repo.get_state(STATE_LAST_HOUR)
    prev_hour_i = int(prev_hour) if prev_hour is not None else -1
    if last_hour_open <= prev_hour_i:
        return last_hour_open

    run_hourly_entries(
        exchange=exchange,
        model_bundle=model_bundle,
        repo=repo,
        main_tf=main_tf,
        main_tf_ms=main_tf_ms,
        btc_last_closed_hour_open_ms=last_hour_open,
        now_ms=_utc_now_ms(),
        main_bars=int(getattr(cfg, "PAPER_MAIN_BARS", 3000)),
        htf_bars=int(getattr(cfg, "PAPER_HTF_BARS", 900)),
        min_main_rows=int(getattr(cfg, "PAPER_MIN_MAIN_ROWS", 400)),
        min_htf_rows=int(getattr(cfg, "PAPER_MIN_HTF_ROWS", 120)),
        execution_runner=execution_runner,
        execution_engine=execution_engine,
        signal_engine=signal_engine,
        candidate_builder=candidate_builder,
        predictor=predictor,
    )
    repo.set_state(STATE_LAST_HOUR, str(last_hour_open))
    logger.info("Часовой тик BTC: обработан бар open_ms=%s", last_hour_open)
    return last_hour_open


def _symbol_has_open(repo: BaseTradesRepository, symbol: str) -> bool:
    for t in repo.list_open_trades(EXEC_TYPE):
        if t.symbol == symbol:
            return True
    return False


def _cooldown_blocks(
    repo: BaseTradesRepository,
    symbol: str,
    *,
    main_tf_ms: int,
    now_ms: int,
    btc_signal_open_ms: int,
) -> bool:
    bars = int(getattr(cfg, "BACKTEST_SL_COOLDOWN_BARS", 0))
    if bars <= 0:
        return False
    last_sl = repo.last_sl_exit_ms(EXEC_TYPE, symbol)
    if last_sl is None:
        return False
    if last_sl + bars * main_tf_ms > btc_signal_open_ms:
        return True
    return False


def _max_sl_day_blocks(repo: BaseTradesRepository, symbol: str, now_ms: int) -> bool:
    cap = int(getattr(cfg, "BACKTEST_MAX_SL_PER_DAY", 0))
    if cap <= 0:
        return False
    day = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    start = int(pd.Timestamp(f"{day}T00:00:00Z").timestamp() * 1000)
    if repo.count_sl_today_utc(EXEC_TYPE, symbol, start, now_ms) >= cap:
        return True
    return False


def run_hourly_entries(
    *,
    exchange,
    model_bundle: ModelBundle,
    repo: BaseTradesRepository,
    main_tf: str,
    main_tf_ms: int,
    btc_last_closed_hour_open_ms: int,
    now_ms: int,
    main_bars: int,
    htf_bars: int,
    min_main_rows: int,
    min_htf_rows: int,
    execution_runner: ExecutionRunner,
    execution_engine: ExecutionEngine,
    signal_engine=None,
    candidate_builder=None,
    predictor=None,
) -> None:
    runtime_predictor = predictor
    scan_symbols = [str(exchange.normalize_symbol(s)) for s in dict.fromkeys(cfg.SYMBOLS)]
    base_map: dict[str, pd.DataFrame] = {}
    htf_map: dict[str, pd.DataFrame] = {}
    htf = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
    skip_reasons: Counter[str] = Counter()

    for symbol in scan_symbols:
        try:
            main_df, hdf = _fetch_symbol_feature_inputs(
                exchange,
                symbol,
                main_tf,
                htf,
                main_bars,
                htf_bars,
            )
        except Exception:
            logger.exception("Загрузка %s", symbol)
            skip_reasons["fetch_error"] += 1
            continue
        if len(main_df) < min_main_rows or len(hdf) < min_htf_rows:
            skip_reasons["insufficient_history"] += 1
            continue
        base_map[symbol] = main_df
        htf_map[symbol] = hdf

    if not base_map:
        logger.warning("Нет символов с данными для часового тика.")
        return

    try:
        pipeline_result = MasterFeatureBuilder().build(base_map, htf_map)
    except Exception:
        logger.exception("Feature pipeline build failed during execution tick")
        return
    use_symbol = "symbol" in model_bundle.feature_names

    portfolio = _build_portfolio_state(repo)

    entry_candidates: list[dict] = []

    for symbol in scan_symbols:
        feat_df = pipeline_result.feature_map.get(symbol)
        if feat_df is None or feat_df.empty:
            skip_reasons["empty_feature_frame"] += 1
            continue

        m = _mask_fully_closed(feat_df, main_tf_ms, now_ms)
        closed_df = feat_df.loc[m]
        if closed_df.empty:
            skip_reasons["no_closed_feature_bar"] += 1
            continue
        last_open_ms = int(_series_open_ms(closed_df["timestamp"].iloc[-1:])[0])
        if last_open_ms != btc_last_closed_hour_open_ms:
            skip_reasons["clock_desync"] += 1
            continue

        with_barriers = etl.attach_barrier_columns(feat_df.copy())
        row = with_barriers.loc[with_barriers["timestamp"] == closed_df["timestamp"].iloc[-1]].iloc[[0]].copy()
        row["symbol"] = symbol

        missing = [f for f in model_bundle.feature_names if f not in row.columns]
        if missing:
            skip_reasons["missing_features"] += 1
            logger.warning("%s: skipped, missing trained features: %s", symbol, ", ".join(missing[:8]))
            continue
        nan_feats = [f for f in model_bundle.feature_names if pd.isna(row[f].iloc[0])]
        if nan_feats:
            if model_bundle.model_type == "lightgbm":
                skip_reasons["nan_features"] += 1
                logger.warning("%s: skipped, NaN features: %s", symbol, ", ".join(nan_feats[:8]))
                continue
            row.loc[:, nan_feats] = 0.0
        if not _is_candidate_event(row, model_bundle.event_filter_config):
            skip_reasons["event_filter"] += 1
            continue
        stop_pct, take_pct = _get_barrier_pcts(row)
        if stop_pct is None or take_pct is None:
            skip_reasons["invalid_barriers"] += 1
            continue

        if _symbol_has_open(repo, symbol):
            skip_reasons["already_open"] += 1
            continue
        if _cooldown_blocks(repo, symbol, main_tf_ms=main_tf_ms, now_ms=now_ms, btc_signal_open_ms=btc_last_closed_hour_open_ms):
            skip_reasons["sl_cooldown"] += 1
            continue
        if _max_sl_day_blocks(repo, symbol, now_ms):
            skip_reasons["sl_day_cap"] += 1
            continue

        if use_symbol and model_bundle.symbol_categories is not None:
            row["symbol"] = pd.Categorical([symbol], categories=model_bundle.symbol_categories)

        prediction = _predict_symbol_proba(
            model_bundle,
            row,
            feat_df,
            predictor=runtime_predictor,
        )
        if prediction is None:
            skip_reasons["prediction_unavailable"] += 1
            continue
        p_short, p_long = prediction

        entry_hour_start = btc_last_closed_hour_open_ms + main_tf_ms
        m1 = _fetch_closed_1m_range(exchange, symbol, entry_hour_start, entry_hour_start + main_tf_ms, exchange.get_timeframe_ms("1m"))
        now_ms2 = _utc_now_ms()
        tf1 = exchange.get_timeframe_ms("1m")
        m1 = m1.loc[_mask_fully_closed(m1, tf1, now_ms2)].reset_index(drop=True)
        if m1.empty:
            logger.debug("%s: нет закрытой 1m для входа (час %s)", symbol, entry_hour_start)
            skip_reasons["missing_entry_minute"] += 1
            continue
        o0 = float(m1.iloc[0]["open"])

        execution_candidate, skip = resolve_and_build_entry_candidate(
            execution_engine,
            symbol=str(symbol),
            next_open=o0,
            stop_pct=float(stop_pct),
            take_pct=float(take_pct),
            p_long=float(p_long),
            p_short=float(p_short),
            balance=float(portfolio.balance),
            risk_per_trade=float(getattr(cfg, "RISK_PER_TRADE", 0.01)),
            allow_longs=bool(getattr(cfg, "ALLOW_LONGS", True)),
            allow_shorts=bool(getattr(cfg, "ALLOW_SHORTS", True)),
            signal_engine=signal_engine,
            candidate_builder=candidate_builder,
        )
        if skip is not None:
            skip_reasons[skip] += 1
            continue

        entry_candidates.append(
            {
                "execution_candidate": execution_candidate,
                "signal_bar_open_ms": btc_last_closed_hour_open_ms,
                "entry_bar_open_ms": entry_hour_start,
                "entry_ts_ms": int(_series_open_ms(m1["timestamp"].iloc[[0]])[0]),
            }
        )

    allocations = execution_runner.select_entries(
        portfolio_state=portfolio,
        positions={str(t.symbol): None for t in repo.list_open_trades(EXEC_TYPE)},
        candidates=[c["execution_candidate"] for c in entry_candidates],
    )
    entry_payload_by_symbol = {
        c["execution_candidate"].symbol: c for c in entry_candidates
    }

    for allocation in allocations:
        c = allocation.candidate
        payload = entry_payload_by_symbol[c.symbol]
        scan_open_ms = int(payload["entry_ts_ms"])
        last_1m_scan = scan_open_ms

        tid = repo.insert_open_trade(
            execution_type=EXEC_TYPE,
            exchange_code=exchange.get_exchange_code(),
            symbol=c.symbol,
            direction=int(c.direction),
            timeframe_signal=main_tf,
            signal_bar_open_ms=int(payload["signal_bar_open_ms"]),
            entry_bar_open_ms=int(payload["entry_bar_open_ms"]),
            entry_ts_ms=int(payload["entry_ts_ms"]),
            entry_price=float(c.entry_price),
            entry_notional=float(allocation.position_notional),
            stop_pct=float(c.stop_pct),
            take_pct=float(c.take_pct),
            p_long=float(c.p_long),
            p_short=float(c.p_short),
            signal_gap=float(c.signal_gap),
            model_name=model_bundle.model_name,
            last_1m_scan_open_ms=last_1m_scan,
            meta={"entry_score": float(c.score)},
        )
        _log_trade_open(
            trade_id=tid,
            symbol=str(c.symbol),
            direction=int(c.direction),
            entry_price=float(c.entry_price),
            notional=float(allocation.position_notional),
            stop_pct=float(c.stop_pct),
            take_pct=float(c.take_pct),
            p_long=float(c.p_long),
            p_short=float(c.p_short),
            signal_gap=float(c.signal_gap),
            score=float(c.score),
            signal_bar_open_ms=int(payload["signal_bar_open_ms"]),
            entry_ts_ms=int(payload["entry_ts_ms"]),
            model_name=model_bundle.model_name,
        )
        if bool(getattr(cfg, "TELEGRAM_NOTIFY_OPEN", True)):
            _send_telegram_message(
                _format_trade_open_message(
                    trade_id=tid,
                    symbol=str(c.symbol),
                    direction=int(c.direction),
                    entry_price=float(c.entry_price),
                    stop_pct=float(c.stop_pct),
                    take_pct=float(c.take_pct),
                    p_long=float(c.p_long),
                    p_short=float(c.p_short),
                    notional=float(allocation.position_notional),
                    model_name=model_bundle.model_name,
                )
            )

    if skip_reasons:
        summary = ", ".join(f"{reason}={count}" for reason, count in skip_reasons.most_common())
        logger.info("Hourly scan skip summary: %s", summary)


def _log_loaded_model(model_bundle: ModelBundle) -> None:
    logger.info(
        "Loaded execution model | name=%s | type=%s | features=%s | seq_len=%s",
        model_bundle.model_name,
        model_bundle.model_type,
        len(model_bundle.feature_names),
        model_bundle.sequence_length,
    )


def _run_tick_cycle(runtime: PaperRuntime) -> None:
    _run_minute_exit_catchup(runtime.repo, runtime.exchange, runtime.execution_runner)
    _run_main_timeframe_cycle(
        repo=runtime.repo,
        model_bundle=runtime.model_bundle,
        exchange=runtime.exchange,
        clock_sym=runtime.clock_sym,
        main_tf=runtime.main_tf,
        main_tf_ms=runtime.main_tf_ms,
        execution_runner=runtime.execution_runner,
        execution_engine=runtime.execution_engine,
        signal_engine=runtime.signal_engine,
        candidate_builder=runtime.candidate_builder,
        predictor=runtime.predictor,
    )


def run_tick(
    *,
    verbose: bool = False,
    exchange=None,
    execution_runner: ExecutionRunner | None = None,
    signal_engine=None,
    candidate_builder=None,
    execution_engine: ExecutionEngine | None = None,
    model=None,
    features_meta: dict | None = None,
    predictor=None,
    model_name: str | None = None,
) -> None:
    _setup_logging(verbose)
    runtime = _build_runtime(
        exchange=exchange,
        execution_runner=execution_runner,
        signal_engine=signal_engine,
        candidate_builder=candidate_builder,
        execution_engine=execution_engine,
        model=model,
        features_meta=features_meta,
        predictor=predictor,
        model_name=model_name,
    )
    _log_loaded_model(runtime.model_bundle)
    _run_tick_cycle(runtime)


def _load_open_trades_full(repo: BaseTradesRepository) -> list[dict]:
    out: list[dict] = []
    for r in repo.list_open_trades(EXEC_TYPE):
        if r.entry_ts_ms is None:
            logger.warning("Пропуск open trade id=%s: отсутствует entry_ts_ms", r.id)
            continue
        out.append(
            {
                "id": int(r.id),
                "symbol": str(r.symbol),
                "direction": int(r.direction),
                "entry_price": float(r.entry_price),
                "entry_notional": float(r.entry_notional),
                "stop_pct": float(r.stop_pct),
                "take_pct": float(r.take_pct),
                "last_1m_scan_open_ms": (
                    int(r.last_1m_scan_open_ms) if r.last_1m_scan_open_ms is not None else None
                ),
                "entry_ts_ms": int(r.entry_ts_ms),
                "repo": repo,
            }
        )
    return out


def cmd_status() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    repo = create_trades_repository_from_config()
    repo.init_schema()
    leverage = float(getattr(cfg, "LEVERAGE", 1))
    initial = float(getattr(cfg, "PAPER_INITIAL_BALANCE", getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)))
    closed_pnl = repo.sum_closed_pnl_quote(EXEC_TYPE)
    margin = repo.sum_open_margin_quote(EXEC_TYPE, leverage)
    wallet = initial + closed_pnl
    available = wallet - margin
    logger.info(
        "paper | wallet=%.2f (initial=%.2f + realized_pnl=%.2f) | margin_open=%.2f | "
        "доступно=%.2f | открыто позиций=%s",
        wallet,
        initial,
        closed_pnl,
        margin,
        available,
        repo.count_open_trades(EXEC_TYPE),
    )
    for t in repo.list_open_trades(EXEC_TYPE):
        logger.info(
            "  OPEN id=%s %s %s entry=%.8g notional=%.2f",
            t.id,
            t.symbol,
            "LONG" if t.direction == 1 else "SHORT",
            t.entry_price,
            t.entry_notional,
        )


def cmd_daemon(
    verbose: bool,
    exchange=None,
    execution_runner: ExecutionRunner | None = None,
    signal_engine=None,
    candidate_builder=None,
    execution_engine: ExecutionEngine | None = None,
    model=None,
    features_meta: dict | None = None,
    predictor=None,
    model_name: str | None = None,
) -> None:
    _setup_logging(verbose)
    runtime = _build_runtime(
        exchange=exchange,
        execution_runner=execution_runner,
        signal_engine=signal_engine,
        candidate_builder=candidate_builder,
        execution_engine=execution_engine,
        model=model,
        features_meta=features_meta,
        predictor=predictor,
        model_name=model_name,
    )
    _log_loaded_model(runtime.model_bundle)
    if runtime.exchange.get_exchange_code() != "bybit":
        raise RuntimeError("Event-driven daemon is currently implemented only for Bybit public kline streams")

    stream = BybitKlineStream()

    try:
        _run_tick_cycle(runtime)
        _sync_ws_topics(stream, runtime.repo, clock_sym=runtime.clock_sym, main_tf=runtime.main_tf)
        stream.start()
        if not stream.wait_until_connected(timeout=15.0):
            raise RuntimeError("Bybit websocket did not connect within 15 seconds")
        _sync_ws_topics(stream, runtime.repo, clock_sym=runtime.clock_sym, main_tf=runtime.main_tf)
        logger.info("Event-driven daemon started | topics are driven by closed kline events")

        main_interval = _ws_interval(runtime.main_tf)
        clock_ws_symbol = _ws_symbol(runtime.clock_sym)

        while True:
            try:
                event = stream.get_event(timeout=60.0)
            except queue.Empty:
                _sync_ws_topics(stream, runtime.repo, clock_sym=runtime.clock_sym, main_tf=runtime.main_tf)
                continue

            if event.interval == "1":
                _handle_1m_kline_close_event(
                    runtime.repo,
                    runtime.exchange,
                    event,
                    tf_1m_ms=runtime.tf_1m_ms,
                    execution_runner=runtime.execution_runner,
                )
                _sync_ws_topics(stream, runtime.repo, clock_sym=runtime.clock_sym, main_tf=runtime.main_tf)
                continue

            if event.interval == main_interval and event.symbol == clock_ws_symbol:
                _run_main_timeframe_cycle(
                    repo=runtime.repo,
                    model_bundle=runtime.model_bundle,
                    exchange=runtime.exchange,
                    clock_sym=runtime.clock_sym,
                    main_tf=runtime.main_tf,
                    main_tf_ms=runtime.main_tf_ms,
                    execution_runner=runtime.execution_runner,
                    execution_engine=runtime.execution_engine,
                    signal_engine=runtime.signal_engine,
                    candidate_builder=runtime.candidate_builder,
                    predictor=runtime.predictor,
                )
                _sync_ws_topics(stream, runtime.repo, clock_sym=runtime.clock_sym, main_tf=runtime.main_tf)
    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C")
        raise
    finally:
        stream.stop()



def main(
    exchange=None,
    execution_runner=None,
    trading_orchestrator=None,
    signal_engine=None,
    risk_manager=None,
    candidate_builder=None,
    feature_runtime_service=None,
    model=None,
    features_meta=None,
    predictor=None,
    model_name=None,
) -> None:
    exchange = exchange or globals().get("exchange")
    execution_runner = execution_runner or EXECUTION_RUNNER

    # Runtime-injected model dependencies (gradual migration path)
    model = model
    features_meta = features_meta
    predictor = predictor
    model_name = model_name
    runtime_predictor = predictor

    # Runtime-injected feature service (gradual migration path)
    runtime_feature_runtime_service = feature_runtime_service

    execution_engine = execution_runner.engine if execution_runner is not None else EXECUTION_ENGINE

    # Runtime-injected core dependencies (gradual migration path)
    trading_orchestrator = trading_orchestrator
    signal_engine = signal_engine
    risk_manager = risk_manager
    candidate_builder = candidate_builder

    p = argparse.ArgumentParser(
        description="Пейпер-трейдинг (журнал execution_trades, sync по BTC).",
        epilog=(
            "Без аргументов — непрерывный режим до Ctrl+C (PAPER_DAEMON_POLL_SEC). "
            "Один проход: python paper.py tick."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "command",
        nargs="?",
        default="daemon",
        choices=("tick", "daemon", "status"),
        help="по умолчанию daemon — цикл до остановки; tick — один проход; status — сводка",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--with-api",
        action="store_true",
        help="Ð¿Ð¾Ð´Ð½ÑÑ‚ÑŒ analytics API Ð² background thread Ð²Ð¼ÐµÑÑ‚Ðµ Ñ bot daemon",
    )
    p.add_argument(
        "--api-host",
        default=None,
        help="host Ð´Ð»Ñ analytics API; Ð¿Ð¾ ÑƒÐ¼Ð¾Ð»Ñ‡Ð°Ð½Ð¸ÑŽ ANALYTICS_API_HOST Ð¸Ð»Ð¸ 0.0.0.0",
    )
    p.add_argument(
        "--api-port",
        type=int,
        default=None,
        help="port Ð´Ð»Ñ analytics API; Ð¿Ð¾ ÑƒÐ¼Ð¾Ð»Ñ‡Ð°Ð½Ð¸ÑŽ ANALYTICS_API_PORT Ð¸Ð»Ð¸ 8001",
    )
    args = p.parse_args()
    if args.command == "status":
        cmd_status()
        return
    if args.command == "daemon":
        _setup_logging(args.verbose)
        if args.with_api:
            start_background_api(host=args.api_host, port=args.api_port)
            logger.info("Analytics API Ð·Ð°Ð¿ÑƒÑ‰ÐµÐ½ Ð² background thread")
        cmd_daemon(
            args.verbose,
            exchange=exchange,
            execution_runner=execution_runner,
            signal_engine=signal_engine,
            candidate_builder=candidate_builder,
            execution_engine=execution_engine,
            model=model,
            features_meta=features_meta,
            predictor=runtime_predictor,
            model_name=model_name,
        )
        return
    if args.with_api:
        logger.warning("--with-api Ð¸Ð¼ÐµÐµÑ‚ ÑÐ¼Ñ‹ÑÐ» Ð´Ð»Ñ command=daemon; Ð´Ð»Ñ %s Ñ„Ð»Ð°Ð³ Ð¸Ð³Ð½Ð¾Ñ€Ð¸Ñ€ÑƒÐµÑ‚ÑÑ", args.command)
    run_tick(
        verbose=args.verbose,
        exchange=exchange,
        execution_runner=execution_runner,
        signal_engine=signal_engine,
        candidate_builder=candidate_builder,
        execution_engine=execution_engine,
        model=model,
        features_meta=features_meta,
        predictor=runtime_predictor,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
