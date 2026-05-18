from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src_refactor.domain.labels.config import ensure_labeling_config

logger = logging.getLogger(__name__)


def get_base_horizon(config: Any) -> int:
    return ensure_labeling_config(config).base_horizon


def compute_effective_horizons(df: pd.DataFrame, config: Any) -> np.ndarray:
    label_config = ensure_labeling_config(config)
    base_horizon = label_config.base_horizon
    if not label_config.enable_adaptive_horizon:
        return np.full(len(df), base_horizon, dtype=np.int32)

    if "realized_vol_1h" not in df.columns:
        logger.warning(
            "Adaptive horizon enabled, but 'realized_vol_1h' is missing. Falling back to fixed horizon=%s.",
            base_horizon,
        )
        return np.full(len(df), base_horizon, dtype=np.int32)

    min_horizon = int(label_config.adaptive_horizon_min or max(1, base_horizon // 2))
    max_horizon = int(label_config.adaptive_horizon_max or base_horizon)
    if min_horizon > max_horizon:
        min_horizon, max_horizon = max_horizon, min_horizon
    min_horizon = max(1, min_horizon)
    max_horizon = max(min_horizon, max_horizon)

    vol_low = float(label_config.adaptive_horizon_vol_low)
    vol_high = float(label_config.adaptive_horizon_vol_high)
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

