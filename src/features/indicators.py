from __future__ import annotations

import numpy as np
import pandas as pd


def _load_pandas_ta():
    try:
        import pandas_ta as ta
    except Exception:
        return None
    return ta


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    ta = _load_pandas_ta()
    if ta is not None:
        atr = ta.atr(high, low, close, length=length)
        if atr is not None:
            return atr

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def compute_linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denominator = np.sum((x - x_mean) ** 2)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        numerator = np.sum((x - x_mean) * (values - y_mean))
        return float(numerator / denominator) if denominator != 0 else np.nan

    return series.rolling(window).apply(slope, raw=True)


def compute_rolling_vwap(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume
    rolling_volume = volume.rolling(window).sum()
    return safe_ratio(price_volume.rolling(window).sum(), rolling_volume)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    ta = _load_pandas_ta()
    if ta is not None:
        adx = ta.adx(high, low, close, length=length)
        if adx is not None and not adx.empty:
            target_col = f"ADX_{length}"
            if target_col in adx.columns:
                return adx[target_col]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * safe_ratio(plus_dm.ewm(alpha=1 / length, adjust=False).mean(), atr)
    minus_di = 100 * safe_ratio(minus_dm.ewm(alpha=1 / length, adjust=False).mean(), atr)
    dx = 100 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def compute_trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    directional_move = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return safe_ratio(directional_move, path_length)
