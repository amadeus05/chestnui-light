from __future__ import annotations

import numpy as np
import pandas as pd


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return safe_ratio(numerator, denominator)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
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
    window = max(1, int(window))
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
    length = max(2, int(length))
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


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    length = max(2, int(length))
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(length)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = safe_ratio(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def matthews_corrcoef_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return float("nan")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float((tp * tn - fp * fn) / denom)


def rolling_matthews_corrcoef_sign_agreement(
    return_a: pd.Series,
    return_b: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(3, window // 2)
    a = return_a.to_numpy(dtype=float)
    b = return_b.to_numpy(dtype=float)
    n = len(a)
    output = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    y_true = (a > 0).astype(np.int64)
    y_pred = (b > 0).astype(np.int64)
    for idx in range(window - 1, n):
        sample = slice(idx - window + 1, idx + 1)
        mask = valid[sample]
        if mask.sum() < min_periods:
            continue
        output[idx] = matthews_corrcoef_binary(y_true[sample][mask], y_pred[sample][mask])
    return pd.Series(output, index=return_a.index)
