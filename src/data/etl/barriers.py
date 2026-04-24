import config as cfg
import numpy as np
import pandas as pd

from src.features.indicators import compute_atr, safe_ratio

from .constants import logger


def get_base_horizon() -> int:
    return int(getattr(cfg, "HORIZON", 16))


def compute_effective_horizons(df: pd.DataFrame) -> np.ndarray:
    base_horizon = max(1, get_base_horizon())
    if not bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)):
        return np.full(len(df), base_horizon, dtype=np.int32)

    if "realized_vol_1h" not in df.columns:
        logger.warning("Adaptive horizon enabled, but 'realized_vol_1h' is missing. Falling back to fixed horizon=%s.", base_horizon)
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
