import config as cfg
import numpy as np
import pandas as pd

from .barriers import compute_effective_horizons, get_base_horizon


def compute_clean_pnl(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    taker_com = float(getattr(cfg, "TAKER_COM", 0.0004))
    return raw_pnl - (taker_com + taker_com)


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
) -> tuple[float | None, str | None]:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)

        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - slippage)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - slippage)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)

        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + slippage)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + slippage)
            return exit_price, "TP"

    return None, None


def simulate_trade_outcome(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    stop_pcts: np.ndarray,
    take_pcts: np.ndarray,
    start_idx: int,
    direction: int,
    horizon: int,
) -> tuple[float, str | None]:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 + slippage) if direction == 1 else base_open * (1 - slippage)
    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if np.isnan(stop_pct) or np.isnan(take_pct):
        return 0.0, None

    for j in range(1, horizon + 1):
        candle_idx = start_idx + j
        if candle_idx >= len(opens):
            break

        exit_price, reason = resolve_trade_exit(
            direction,
            entry_price,
            opens[candle_idx],
            highs[candle_idx],
            lows[candle_idx],
            stop_pct,
            take_pct,
        )
        if exit_price is not None:
            return compute_clean_pnl(direction, entry_price, exit_price), reason

    return 0.0, None


def triple_barrier_labeling(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    effective_horizons = compute_effective_horizons(df)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else get_base_horizon()

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - max_horizon):
        label = 0
        horizon = int(effective_horizons[i]) if i < len(effective_horizons) else get_base_horizon()
        long_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=1, horizon=horizon)
        short_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=-1, horizon=horizon)

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1

        labels.append(label)

    labels.extend([0] * max_horizon)
    output = df.copy()
    output["Target"] = labels
    return output
