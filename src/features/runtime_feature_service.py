from __future__ import annotations

import logging

import config as cfg
import numpy as np
import pandas as pd

from src.features.indicators import compute_atr, safe_ratio

logger = logging.getLogger(__name__)

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]


def get_base_horizon() -> int:
    return int(getattr(cfg, "HORIZON", 16))


def compute_effective_horizons(df: pd.DataFrame) -> np.ndarray:
    base_horizon = max(1, get_base_horizon())
    if not bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)):
        return np.full(len(df), base_horizon, dtype=np.int32)

    if "realized_vol_1h" not in df.columns:
        logger.warning(
            "Adaptive horizon enabled, but 'realized_vol_1h' is missing. Falling back to fixed horizon=%s.",
            base_horizon,
        )
        return np.full(len(df), base_horizon, dtype=np.int32)

    min_horizon = int(getattr(cfg, "ADAPTIVE_HORIZON_MIN", max(1, base_horizon // 2)))
    max_horizon = int(getattr(cfg, "ADAPTIVE_HORIZON_MAX", base_horizon))
    if min_horizon > max_horizon:
        min_horizon, max_horizon = max_horizon, min_horizon
    min_horizon = max(1, min_horizon)
    max_horizon = max(min_horizon, max_horizon)

    vol_low = float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.005))
    vol_high = float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.025))
    if not np.isfinite(vol_low) or not np.isfinite(vol_high) or vol_high <= vol_low:
        logger.warning(
            "Invalid adaptive horizon volatility bounds (low=%s, high=%s). Falling back to fixed horizon=%s.",
            vol_low,
            vol_high,
            base_horizon,
        )
        return np.full(len(df), base_horizon, dtype=np.int32)

    vol = pd.Series(df["realized_vol_1h"], copy=False).astype(float).abs()
    normalized = ((vol - vol_low) / (vol_high - vol_low)).clip(lower=0.0, upper=1.0)
    normalized_values = normalized.to_numpy()
    adaptive_raw = np.rint(max_horizon - normalized_values * (max_horizon - min_horizon))
    adaptive = np.full(len(df), base_horizon, dtype=np.int32)
    valid_mask = np.isfinite(adaptive_raw)
    adaptive[valid_mask] = np.clip(adaptive_raw[valid_mask], min_horizon, max_horizon).astype(np.int32)
    return adaptive


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
    realized_vol_1h: pd.Series,
    effective_horizons: np.ndarray | None = None,
) -> pd.Series:
    atr_pct = safe_ratio(atr_14, close).abs()
    if effective_horizons is None:
        horizon_sqrt = np.sqrt(float(get_base_horizon()))
    else:
        horizon_sqrt = np.sqrt(np.maximum(effective_horizons.astype(float), 1.0))
    horizon_vol_pct = realized_vol_1h.abs() * horizon_sqrt

    stop_pct = pd.concat(
        [
            atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25)),
            horizon_vol_pct * float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)),
        ],
        axis=1,
    ).max(axis=1)

    min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", getattr(cfg, "SL_PCT", 0.015)))
    max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", getattr(cfg, "TP_PCT", 0.03)))
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
    return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))


def attach_barrier_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)
    effective_horizons = compute_effective_horizons(output)

    if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", True)):
        if "realized_vol_1h" not in output.columns:
            raise ValueError("Dynamic barriers require feature 'realized_vol_1h' to be enabled.")
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(
            close,
            atr_14,
            output["realized_vol_1h"],
            effective_horizons=effective_horizons,
        )
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"])
    else:
        output["barrier_stop_pct"] = float(getattr(cfg, "SL_PCT", 0.015))
        output["barrier_take_pct"] = float(getattr(cfg, "TP_PCT", 0.03))
    return output


def build_candle_maps(
    repository,
    symbols_to_load: list,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
        htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        df = repository.load_candles(symbol, timeframe)
        htf_df = repository.load_candles(symbol, htf_timeframe)
        funding_df = repository.load_funding_rates(symbol)
        premium_index_df = repository.load_premium_index_klines(symbol, timeframe)
        open_interest_df = repository.load_open_interest(symbol, timeframe)
        if df.empty or htf_df.empty:
            logger.warning("%s: no data in DB (main=%s, htf=%s)", symbol_name, len(df), len(htf_df))
            continue

        df = attach_funding_context(df, funding_df)
        df = attach_premium_index_context(df, premium_index_df)
        df = attach_open_interest_context(df, open_interest_df)
        logger.info(
            "%s: main=%s, htf=%s rows, funding=%s points, premium=%s points, open_interest=%s points",
            symbol_name,
            len(df),
            len(htf_df),
            len(funding_df),
            len(premium_index_df),
            len(open_interest_df),
        )
        base_candle_map[symbol_name] = df
        htf_candle_map[symbol_name] = htf_df

    return base_candle_map, htf_candle_map


def attach_funding_context(
    base_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if funding_df is None or funding_df.empty:
        output["funding_rate"] = np.nan
        return output

    funding_frame = funding_df[["timestamp", "funding_rate"]].copy().sort_values("timestamp").reset_index(drop=True)
    return pd.merge_asof(
        output,
        funding_frame,
        on="timestamp",
        direction="backward",
    )


def attach_premium_index_context(
    base_df: pd.DataFrame,
    premium_index_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if premium_index_df is None or premium_index_df.empty:
        output["premium_index_close"] = np.nan
        return output

    premium_frame = (
        premium_index_df[["timestamp", "premium_index_close"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return pd.merge_asof(
        output,
        premium_frame,
        on="timestamp",
        direction="backward",
    )


def attach_open_interest_context(
    base_df: pd.DataFrame,
    open_interest_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if open_interest_df is None or open_interest_df.empty:
        output["open_interest"] = np.nan
        return output

    open_interest_frame = (
        open_interest_df[["timestamp", "open_interest"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return pd.merge_asof(
        output,
        open_interest_frame,
        on="timestamp",
        direction="backward",
    )
