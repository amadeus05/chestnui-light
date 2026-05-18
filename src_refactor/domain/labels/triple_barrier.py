from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src_refactor.domain.execution import (
    ExecutionPricingConfig,
    apply_entry_slippage,
    compute_net_pnl_pct,
    resolve_trade_exit,
)
from src_refactor.domain.labels.config import ensure_labeling_config
from src_refactor.domain.labels.horizons import compute_effective_horizons, get_base_horizon


def simulate_label_trade_outcome(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    stop_pcts: np.ndarray,
    take_pcts: np.ndarray,
    start_idx: int,
    direction: int,
    horizon: int,
    *,
    config: Any,
) -> tuple[float, str | None]:
    label_config = ensure_labeling_config(config)
    pricing = ExecutionPricingConfig(taker_fee=label_config.taker_fee, slippage=label_config.slippage)
    base_open = opens[start_idx + 1]
    entry_price = apply_entry_slippage(direction, base_open, pricing)
    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if np.isnan(stop_pct) or np.isnan(take_pct):
        return 0.0, None

    for j in range(1, horizon + 1):
        candle_idx = start_idx + j
        if candle_idx >= len(opens):
            break

        exit_result = resolve_trade_exit(
            direction,
            entry_price,
            opens[candle_idx],
            highs[candle_idx],
            lows[candle_idx],
            stop_pct,
            take_pct,
            pricing,
        )
        if exit_result.price is not None:
            return (
                compute_net_pnl_pct(
                    direction,
                    entry_price,
                    exit_result.price,
                    pricing,
                ),
                exit_result.reason,
            )

    return 0.0, None


def triple_barrier_labeling(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    label_config = ensure_labeling_config(config)
    labels = []
    effective_horizons = compute_effective_horizons(df, label_config)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else get_base_horizon(label_config)

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - max_horizon):
        label = 0
        horizon = int(effective_horizons[i]) if i < len(effective_horizons) else get_base_horizon(label_config)
        long_pnl, _ = simulate_label_trade_outcome(
            opens,
            highs,
            lows,
            stop_pcts,
            take_pcts,
            i,
            direction=1,
            horizon=horizon,
            config=label_config,
        )
        short_pnl, _ = simulate_label_trade_outcome(
            opens,
            highs,
            lows,
            stop_pcts,
            take_pcts,
            i,
            direction=-1,
            horizon=horizon,
            config=label_config,
        )

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1

        labels.append(label)

    labels.extend([0] * max_horizon)
    output = df.copy()
    output["Target"] = labels
    return output

