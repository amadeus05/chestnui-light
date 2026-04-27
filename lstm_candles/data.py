from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
import etl
import train
from src.features.indicators import compute_atr
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


SEQUENCE_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_vs_ema_20",
    "close_vs_ema_50",
    "ema20_slope_3",
    "atr14_norm",
    "realized_vol_24",
    "volume_norm_24",
    "funding_rate",
    "funding_rate_change_8h",
    "premium_index_close",
    "premium_index_change_8h",
    "open_interest_change_8h",
    "btc_return_1h",
    "asset_minus_btc_return_1h",
]


def _apply_end_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    end_cutoff = train.get_end_date_cutoff()
    if end_cutoff is None or pd.isna(end_cutoff):
        return frame
    return frame.loc[frame[train.TIMESTAMP_COLUMN] <= end_cutoff].copy()


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator.astype(float) / denominator.astype(float)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio > 0)
    return np.log(ratio)


def _pct_delta(series: pd.Series, periods: int) -> pd.Series:
    base = series.shift(periods).replace(0, np.nan)
    return ((series - base) / base).replace([np.inf, -np.inf], np.nan)


def _build_btc_context(repository: HistoricalKlineRepository) -> pd.DataFrame:
    btc_symbol = "BTC/USDT"
    btc = repository.load_candles(btc_symbol, cfg.TIMEFRAME)
    btc = _apply_end_cutoff(btc)
    if btc is None or btc.empty:
        return pd.DataFrame(columns=[train.TIMESTAMP_COLUMN, "btc_return_1h"])

    btc = btc.sort_values(train.TIMESTAMP_COLUMN).reset_index(drop=True).copy()
    btc["btc_return_1h"] = _safe_log_ratio(btc["close"], btc["close"].shift(1))
    return btc[[train.TIMESTAMP_COLUMN, "btc_return_1h"]].copy()


def build_symbol_sequence_frame(
    repository: HistoricalKlineRepository,
    symbol: str,
    btc_context: pd.DataFrame,
) -> pd.DataFrame:
    main = repository.load_candles(symbol, cfg.TIMEFRAME)
    if main is None or main.empty:
        return pd.DataFrame(columns=[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN] + SEQUENCE_FEATURE_COLUMNS)

    main = _apply_end_cutoff(main)
    if main is None or main.empty:
        return pd.DataFrame(columns=[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN] + SEQUENCE_FEATURE_COLUMNS)

    funding = _apply_end_cutoff(repository.load_funding_rates(symbol))
    premium = _apply_end_cutoff(repository.load_premium_index_klines(symbol, cfg.TIMEFRAME))
    open_interest = _apply_end_cutoff(repository.load_open_interest(symbol, cfg.TIMEFRAME))

    frame = main.sort_values(train.TIMESTAMP_COLUMN).reset_index(drop=True).copy()
    frame = etl.attach_funding_context(frame, funding)
    frame = etl.attach_premium_index_context(frame, premium)
    frame = etl.attach_open_interest_context(frame, open_interest)

    if btc_context is not None and not btc_context.empty:
        frame = pd.merge_asof(
            frame.sort_values(train.TIMESTAMP_COLUMN),
            btc_context.sort_values(train.TIMESTAMP_COLUMN),
            on=train.TIMESTAMP_COLUMN,
            direction="backward",
        )
    else:
        frame["btc_return_1h"] = np.nan

    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    funding_rate = pd.to_numeric(frame.get("funding_rate"), errors="coerce")
    premium_index = pd.to_numeric(frame.get("premium_index_close"), errors="coerce")
    open_interest_series = pd.to_numeric(frame.get("open_interest"), errors="coerce")

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    atr14 = compute_atr(high, low, close, length=14)
    log_return_1 = _safe_log_ratio(close, close.shift(1))

    output = pd.DataFrame({train.TIMESTAMP_COLUMN: frame[train.TIMESTAMP_COLUMN].copy()})
    output["log_return_1"] = log_return_1
    output["log_return_4"] = _safe_log_ratio(close, close.shift(4))
    output["body_pct"] = ((close - open_) / open_).replace([np.inf, -np.inf], np.nan)
    output["range_pct"] = ((high - low) / open_).replace([np.inf, -np.inf], np.nan)
    output["upper_wick_pct"] = ((high - np.maximum(open_, close)) / open_).replace([np.inf, -np.inf], np.nan)
    output["lower_wick_pct"] = ((np.minimum(open_, close) - low) / open_).replace([np.inf, -np.inf], np.nan)
    output["close_vs_ema_20"] = ((close - ema20) / close).replace([np.inf, -np.inf], np.nan)
    output["close_vs_ema_50"] = ((close - ema50) / close).replace([np.inf, -np.inf], np.nan)
    output["ema20_slope_3"] = _pct_delta(ema20, 3)
    output["atr14_norm"] = (atr14 / close).replace([np.inf, -np.inf], np.nan)
    output["realized_vol_24"] = log_return_1.rolling(24).std()
    volume_mean_24 = volume.rolling(24).mean().replace(0, np.nan)
    output["volume_norm_24"] = _safe_log_ratio(volume, volume_mean_24)
    output["funding_rate"] = funding_rate.fillna(0.0)
    output["funding_rate_change_8h"] = (funding_rate - funding_rate.shift(8)).fillna(0.0)
    output["premium_index_close"] = premium_index.fillna(0.0)
    output["premium_index_change_8h"] = (premium_index - premium_index.shift(8)).fillna(0.0)
    output["open_interest_change_8h"] = _pct_delta(open_interest_series, 8).fillna(0.0)
    output["btc_return_1h"] = pd.to_numeric(frame.get("btc_return_1h"), errors="coerce").fillna(0.0)
    output["asset_minus_btc_return_1h"] = (output["log_return_1"] - output["btc_return_1h"]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    output[train.SYMBOL_COLUMN] = symbol

    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    return output[[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN] + SEQUENCE_FEATURE_COLUMNS].copy()


def load_candle_sequence_frames(db_path: str, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    repository = HistoricalKlineRepository(db_path=db_path)
    btc_context = _build_btc_context(repository)
    frames = []
    for symbol in symbols:
        symbol_frame = build_symbol_sequence_frame(repository, symbol, btc_context)
        if not symbol_frame.empty:
            frames.append(symbol_frame)

    if not frames:
        raise RuntimeError("No candle-sequence frames could be built. Check raw candles/context tables.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values([train.SYMBOL_COLUMN, train.TIMESTAMP_COLUMN]).reset_index(drop=True)
    return dataset, list(SEQUENCE_FEATURE_COLUMNS)

