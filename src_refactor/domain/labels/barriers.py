from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src_refactor.domain.features.indicators import compute_atr, safe_ratio
from src_refactor.domain.labels.config import ensure_labeling_config
from src_refactor.domain.labels.horizons import compute_effective_horizons


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
    realized_vol_1h: pd.Series,
    *,
    config: Any,
    effective_horizons: np.ndarray | None = None,
) -> pd.Series:
    label_config = ensure_labeling_config(config)
    atr_pct = safe_ratio(atr_14, close).abs()
    if effective_horizons is None:
        horizon_sqrt = np.sqrt(float(label_config.base_horizon))
    else:
        horizon_sqrt = np.sqrt(np.maximum(effective_horizons.astype(float), 1.0))
    horizon_vol_pct = realized_vol_1h.abs() * horizon_sqrt

    stop_pct = pd.concat(
        [
            atr_pct * float(label_config.barrier_atr_multiplier),
            horizon_vol_pct * float(label_config.barrier_rvol_multiplier),
        ],
        axis=1,
    ).max(axis=1)

    min_pct = float(label_config.barrier_min_pct if label_config.barrier_min_pct is not None else label_config.sl_pct)
    max_pct = float(label_config.barrier_max_pct if label_config.barrier_max_pct is not None else label_config.tp_pct)
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series, config: Any) -> pd.Series:
    return stop_pct * float(ensure_labeling_config(config).barrier_tp_to_sl_ratio)


def attach_barrier_columns(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    label_config = ensure_labeling_config(config)
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)
    effective_horizons = compute_effective_horizons(output, config)

    if label_config.use_dynamic_barriers:
        if "realized_vol_1h" not in output.columns:
            raise ValueError("Dynamic barriers require feature 'realized_vol_1h' to be enabled.")
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(
            close,
            atr_14,
            output["realized_vol_1h"],
            config=label_config,
            effective_horizons=effective_horizons,
        )
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"], label_config)
    else:
        output["barrier_stop_pct"] = float(label_config.sl_pct)
        output["barrier_take_pct"] = float(label_config.tp_pct)
    return output

