"""
Загрузка свечей/фич и утилиты бэктеста без глобального import *.
Используется из backtest_engine, portfolio_backtest_runner и портфельного режима TradingEngine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from signal_filter import build_candidate_event_mask
from src.data.etl.barriers import attach_barrier_columns
from src.data.etl.contexts import (
    attach_funding_context,
    attach_open_interest_context,
    attach_premium_index_context,
)
from src.features import FeaturePipeline
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

DEFAULT_MODEL_NAME = "lightgbm_target"


def get_end_date_cutoff() -> pd.Timestamp | None:
    end = getattr(cfg, "END_DATE", None)
    if not end:
        return None
    return pd.to_datetime(end, errors="coerce")


def apply_end_date_cutoff(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or timestamp_column not in df.columns:
        return df
    end_cutoff = get_end_date_cutoff()
    if end_cutoff is None or pd.isna(end_cutoff):
        return df
    return df.loc[df[timestamp_column] <= end_cutoff].copy()


def parse_period_payload(period_payload: dict | None, period_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not period_payload:
        raise RuntimeError(
            f"Model metadata does not include {period_name}. Re-run train.py with holdout-aware artifacts first."
        )
    start = pd.to_datetime(period_payload.get("start"), errors="coerce")
    end = pd.to_datetime(period_payload.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise RuntimeError(f"Model metadata has an invalid {period_name}: {period_payload}")
    if start > end:
        raise RuntimeError(f"Model metadata has {period_name} start after end: {period_payload}")
    return start, end


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TF_MS[timeframe]


def load_raw_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    repository = HistoricalKlineRepository()
    df = repository.load_candles(symbol, timeframe)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    tf_ms = timeframe_to_ms(timeframe)
    df["close_time"] = df["timestamp"] + pd.to_timedelta(tf_ms, unit="ms")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = apply_end_date_cutoff(df)
    return df


def _enrich_main_ohlcv(repository: HistoricalKlineRepository, symbol: str, main_df: pd.DataFrame) -> pd.DataFrame:
    funding_df = repository.load_funding_rates(symbol)
    premium_index_df = repository.load_premium_index_klines(symbol, str(cfg.TIMEFRAME))
    open_interest_df = repository.load_open_interest(symbol, str(cfg.TIMEFRAME))
    df = main_df.copy()
    df = attach_funding_context(df, funding_df)
    df = attach_premium_index_context(df, premium_index_df)
    df = attach_open_interest_context(df, open_interest_df)
    return df


def load_all_raw_data(symbols: list[str]) -> dict:
    all_data: dict = {}
    repository = HistoricalKlineRepository()
    realtime = bool(getattr(cfg, "BACKTEST_REALTIME_FEATURES", False))

    print(f"Loading raw data for {len(symbols)} symbols...")
    for sym in symbols:
        try:
            df_main = load_raw_candles(sym, str(cfg.TIMEFRAME))
            df_htf = load_raw_candles(sym, str(cfg.HTF_TIMEFRAME))
            if df_main.empty:
                print(f"Warning: {sym} has no main TF data ({cfg.TIMEFRAME})")
                continue
            if df_htf.empty:
                print(f"Warning: {sym} has no HTF data ({cfg.HTF_TIMEFRAME})")
                continue
            main = df_main
            if realtime:
                main = _enrich_main_ohlcv(repository, sym, main)
            all_data[sym] = {"main": main, "htf": df_htf}
            print(f"{sym}: main={len(main)} candles, htf={len(df_htf)} candles")
        except Exception as e:
            print(f"Warning: failed loading {sym}: {e}")
    return all_data


def load_precomputed_features(
    symbol: str,
    symbol_categories=None,
    required_columns: list | None = None,
) -> pd.DataFrame:
    repository = HistoricalKlineRepository()
    try:
        df = repository.load_features(symbol)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if required_columns:
        required_order = list(dict.fromkeys(["timestamp"] + required_columns + ["symbol"]))
        missing_columns = [column for column in required_order if column not in df.columns]
        if missing_columns:
            return pd.DataFrame()
        df = df[required_order].copy()
    df = apply_end_date_cutoff(df)
    if symbol_categories is None:
        df["symbol"] = df["symbol"].astype("category")
    else:
        df["symbol"] = pd.Categorical(df["symbol"], categories=symbol_categories)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_timestamp_index(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    return df.set_index("timestamp", drop=False).to_dict("index")


def get_common_main_timestamps(all_data: dict) -> list:
    if not all_data:
        return []
    ts_sets = [set(payload["main"]["timestamp"]) for payload in all_data.values()]
    return sorted(list(set.intersection(*ts_sets))) if ts_sets else []


def filter_symbols_with_recent_data(all_data: dict, min_common_candles: int = 300) -> tuple[dict, list]:
    if not all_data:
        return {}, []
    timeframe_delta = pd.to_timedelta(timeframe_to_ms(str(cfg.TIMEFRAME)), unit="ms")
    latest_timestamp = max(payload["main"]["timestamp"].iloc[-1] for payload in all_data.values())
    recent_cutoff = latest_timestamp - (timeframe_delta * min_common_candles)
    filtered = {}
    dropped = []
    for symbol, payload in all_data.items():
        symbol_last_ts = payload["main"]["timestamp"].iloc[-1]
        if symbol_last_ts < recent_cutoff:
            dropped.append(symbol)
            continue
        filtered[symbol] = payload
    return filtered, dropped


def filter_symbols_with_period_overlap(
    all_data: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    min_candles: int = 2,
) -> tuple[dict, list]:
    if not all_data:
        return {}, []
    filtered = {}
    dropped = []
    for symbol, payload in all_data.items():
        period_rows = payload["main"].loc[
            (payload["main"]["timestamp"] >= start_ts) & (payload["main"]["timestamp"] <= end_ts)
        ]
        if len(period_rows) < min_candles:
            dropped.append(symbol)
            continue
        filtered[symbol] = payload
    return filtered, dropped


def prepare_dataset_for_time(df: pd.DataFrame, analysis_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df[df["close_time"] <= analysis_ts].copy()
    if out.empty:
        return out
    return out.drop(columns=["close_time"], errors="ignore").reset_index(drop=True)


def build_feature_row_at_time(
    all_raw: dict,
    target_symbol: str,
    analysis_ts: pd.Timestamp,
    feature_names: list,
    symbol_categories=None,
):
    base_map: dict[str, pd.DataFrame] = {}
    htf_map: dict[str, pd.DataFrame] = {}
    for sym, payload in all_raw.items():
        main_cut = prepare_dataset_for_time(payload["main"], analysis_ts)
        htf_cut = prepare_dataset_for_time(payload["htf"], analysis_ts)
        if main_cut is None or main_cut.empty or len(main_cut) < 250:
            continue
        if htf_cut is None or htf_cut.empty or len(htf_cut) < 60:
            continue
        base_map[sym] = main_cut
        htf_map[sym] = htf_cut
    if target_symbol not in base_map:
        return None
    try:
        pipeline = FeaturePipeline()
        result = pipeline.compute(base_map, htf_map)
        feat_df = result.feature_map.get(target_symbol)
        if feat_df is None or feat_df.empty:
            return None
        feat_df = attach_barrier_columns(feat_df)
        feat_df = feat_df.copy()
        feat_df["symbol"] = target_symbol
        if symbol_categories is None:
            feat_df["symbol"] = feat_df["symbol"].astype("category")
        else:
            feat_df["symbol"] = pd.Categorical(feat_df["symbol"], categories=symbol_categories)
        latest_row = feat_df.iloc[[-1]].copy()
        missing = [f for f in feature_names if f not in latest_row.columns]
        if missing:
            print(f"Warning: {target_symbol} missing features: {missing[:10]}")
            return None
        if latest_row[feature_names].isna().any(axis=None):
            return None
        return latest_row
    except Exception as e:
        print(f"Warning: feature build failed for {target_symbol} @ {analysis_ts}: {e}")
        return None


def get_exec_row_by_ts(df: pd.DataFrame, ts: pd.Timestamp):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None
    return row.iloc[0]


def get_feature_row_precomputed(df: pd.DataFrame, ts: pd.Timestamp, feature_names: list):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None
    latest_row = row.iloc[[-1]].copy()
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None
    if latest_row[feature_names].isna().any(axis=None):
        return None
    return latest_row


def get_exec_row_by_ts_index(indexed_rows: dict, ts: pd.Timestamp):
    return indexed_rows.get(ts)


def get_feature_row_precomputed_index(indexed_rows: dict, ts: pd.Timestamp, feature_names: list):
    row = indexed_rows.get(ts)
    if row is None:
        return None
    latest_row = pd.DataFrame([row])
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None
    if latest_row[feature_names].isna().any(axis=None):
        return None
    return latest_row


def is_candidate_event(feature_row: pd.DataFrame, event_filter_config: dict) -> bool:
    if feature_row is None or feature_row.empty:
        return False
    mask = build_candidate_event_mask(feature_row, event_filter_config)
    return bool(mask.iloc[0]) if not mask.empty else False


def normalize_features_for_model(latest_row: pd.DataFrame, feature_names: list, symbol_categories=None):
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict):
    if not clip_bounds:
        return frame
    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def prepare_precomputed_feature_store(
    feat_df: pd.DataFrame,
    feature_names: list,
    symbol_categories=None,
    clip_bounds: dict | None = None,
) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return pd.DataFrame(columns=feature_names)
    missing = [feature for feature in feature_names if feature not in feat_df.columns]
    if missing:
        return pd.DataFrame(columns=feature_names)
    timestamps = feat_df["timestamp"].copy()
    prepared = normalize_features_for_model(feat_df, feature_names, symbol_categories=symbol_categories)
    prepared = apply_feature_clip_bounds(prepared, clip_bounds or {})
    prepared.insert(0, "timestamp", timestamps.values)
    prepared = prepared.dropna(subset=feature_names)
    prepared = prepared.drop_duplicates(subset=["timestamp"], keep="last")
    return prepared.set_index("timestamp", drop=True).sort_index()


def get_feature_batch_precomputed(feature_store: dict, symbols: list, ts: pd.Timestamp):
    batch_frames = []
    batch_symbols = []
    for symbol in symbols:
        prepared = feature_store.get(symbol)
        if prepared is None or prepared.empty or ts not in prepared.index:
            continue
        batch_frames.append(prepared.loc[[ts]])
        batch_symbols.append(symbol)
    if not batch_frames:
        return [], pd.DataFrame()
    return batch_symbols, pd.concat(batch_frames, axis=0)


def build_prediction_lookup(predictions: pd.DataFrame | None) -> dict:
    if predictions is None or predictions.empty:
        return {}
    required_columns = {"timestamp", "symbol", "p_short", "p_long"}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Walk-forward predictions missing columns: {missing_columns}")
    prepared = predictions.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared = prepared.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return {
        (row.timestamp, row.symbol): (float(row.p_short), float(row.p_long))
        for row in prepared.itertuples(index=False)
    }


def get_barrier_pcts(feature_row: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None
    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None
    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct
