import logging

import config as cfg
import numpy as np
import pandas as pd
import pandas_ta as ta

from config import *
from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_service import BinanceService
from src.exchanges.bybit.bybit_service import BybitService
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_ZSCORE_WINDOW = max(10, int(getattr(cfg, "MARKET_ZSCORE_WINDOW", 96)))
EMA_FAST_WINDOW = max(2, int(getattr(cfg, "EMA_FAST_WINDOW", 12)))
EMA_SLOW_WINDOW = max(EMA_FAST_WINDOW + 1, int(getattr(cfg, "EMA_SLOW_WINDOW", 48)))
EMA_SLOPE_BASE_WINDOW_4H = max(2, int(getattr(cfg, "EMA_SLOPE_BASE_WINDOW_4H", 21)))
EMA_SLOPE_WINDOW_4H = max(2, int(getattr(cfg, "EMA_SLOPE_WINDOW_4H", 6)))
REALIZED_VOL_WINDOW_1H = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_1H", 24)))
REALIZED_VOL_WINDOW_4H = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_4H", 20)))
RANGE_WINDOW_4H = max(2, int(getattr(cfg, "RANGE_WINDOW_4H", 14)))
VWAP_WINDOW_4H = max(2, int(getattr(cfg, "VWAP_WINDOW_4H", 20)))
CHOPPINESS_WINDOW_1H = max(2, int(getattr(cfg, "CHOPPINESS_WINDOW_1H", 14)))
PRICE_ACTION_LEVEL_WINDOW_1H = max(4, int(getattr(cfg, "PRICE_ACTION_LEVEL_WINDOW_1H", 24)))
RANGE_COMPRESSION_SHORT_WINDOW_1H = max(2, int(getattr(cfg, "RANGE_COMPRESSION_SHORT_WINDOW_1H", 12)))
RANGE_COMPRESSION_LONG_WINDOW_1H = max(
    RANGE_COMPRESSION_SHORT_WINDOW_1H + 1,
    int(getattr(cfg, "RANGE_COMPRESSION_LONG_WINDOW_1H", 48)),
)

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FEATURE_OUTPUT_COLUMNS = [
    "return_1h_6",
    "return_1h_12",
    "return_1h_24",
    "realized_vol_1h",
    "ema_fast_slow",
    "linear_regression_slope_atr_1h_12",
    "linear_regression_slope_atr_1h_24",
    "volatility_regime_change_1h",
    "return_4h_1",
    "return_4h_3",
    "return_4h_7",
    "return_4h_14",
    "ema_slope_4h",
    "atr_ratio_1h",
    "realized_vol_4h_returns_20",
    "zscore_vs_vwap_4h",
    "vol_ratio",
    # "parkinson_ratio",  # beta disabled
    # "choppiness_index",  # beta disabled
    # "volume_imbalance_1h_12",  # beta disabled
    "price_position_1h",
    "distance_to_support_1h",
    "distance_to_resistance_1h",
    "price_position_4h",
    "adx_4h",
    "distance_to_rolling_high_4h",
    "distance_to_rolling_low_4h",
    "cross_sectional_rank_4h",
]
# Test features
TEST_FEATURE_COLUMNS = [
    # Range compression
    "range_compression_1h",
    # Session positioning
    "distance_to_session_high_1h",
    "distance_to_session_low_1h",
    # Trend persistence / acceleration
    "trend_persistence_score_12",
    "trend_persistence_score_24",
    # Trend efficiency
    "trend_efficiency_24h",
    # Acceleration
    "slope_acceleration_1h_12_24",
    "ema_slope_acceleration_1h",
    "volatility_acceleration_1h",
    # Time context
    "hour_sin_1h",
    "hour_cos_1h",
    "is_weekend_1h",
    # Asset-specific alpha
    "relative_strength_vs_btc_24h",
    "beta_to_btc_24h",
    "residual_return_24h",
    # Cross-sectional context
    "cross_sectional_rank_ema_fast_slow_1h",
    # Market context
    "market_breadth_ema_fast_slow_1h",
    "market_breadth_pos_return_4h_3",
    "market_dispersion_return_4h_3",
    "delta_market_breadth_ema_fast_slow_1h",
    "market_breadth_ema_fast_slow_1h_zscore",
    "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
    "trend_efficiency_24h_x_volatility_regime_change_1h",
]
if bool(getattr(cfg, "ENABLE_TEST_FEATURES", False)):
    FEATURE_OUTPUT_COLUMNS.extend(TEST_FEATURE_COLUMNS)
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]
OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + FEATURE_OUTPUT_COLUMNS + BARRIER_OUTPUT_COLUMNS + ["Target"]


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def compute_atr(high, low, close, length):
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


def compute_linear_regression_slope(series, window):
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denominator = np.sum((x - x_mean) ** 2)

    def slope(values):
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        numerator = np.sum((x - x_mean) * (values - y_mean))
        return numerator / denominator if denominator != 0 else np.nan

    return series.rolling(window).apply(slope, raw=True)


def compute_rolling_vwap(close, high, low, volume, window):
    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume
    rolling_volume = volume.rolling(window).sum()
    return safe_ratio(price_volume.rolling(window).sum(), rolling_volume)


def compute_adx(high, low, close, length=14):
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


def compute_dynamic_barrier_stop_pct(close, atr_14, realized_vol_1h):
    atr_pct = safe_ratio(atr_14, close).abs()
    horizon_vol_pct = realized_vol_1h.abs() * np.sqrt(HORIZON)

    stop_pct = pd.concat(
        [
            atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25)),
            horizon_vol_pct * float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)),
        ],
        axis=1,
    ).max(axis=1)

    min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", SL_PCT))
    max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", TP_PCT))
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct):
    return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))


def normalize_rank(series):
    if len(series) == 1:
        return pd.Series(0.5, index=series.index)
    ranked = series.rank(method="average")
    return (ranked - 1) / (len(series) - 1)


def compute_trend_efficiency(close, window):
    directional_move = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return safe_ratio(directional_move, path_length)


def add_test_features(df, atr_14, ema_fast_slow, atr_ratio_1h):
    """Test features block."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    range_short_1h = high.rolling(RANGE_COMPRESSION_SHORT_WINDOW_1H).max() - low.rolling(RANGE_COMPRESSION_SHORT_WINDOW_1H).min()
    range_long_1h = high.rolling(RANGE_COMPRESSION_LONG_WINDOW_1H).max() - low.rolling(RANGE_COMPRESSION_LONG_WINDOW_1H).min()
    df["range_compression_1h"] = safe_ratio(range_short_1h, range_long_1h)

    session_key = df["timestamp"].dt.floor("D")
    session_high_1h = high.groupby(session_key).cummax()
    session_low_1h = low.groupby(session_key).cummin()
    df["distance_to_session_high_1h"] = safe_ratio(session_high_1h - close, atr_14)
    df["distance_to_session_low_1h"] = safe_ratio(close - session_low_1h, atr_14)

    signed_step = np.sign(close.diff())
    df["trend_persistence_score_12"] = signed_step.rolling(12).mean()
    df["trend_persistence_score_24"] = signed_step.rolling(24).mean()

    df["trend_efficiency_24h"] = compute_trend_efficiency(close, 24)
    slope_12 = compute_linear_regression_slope(close, 12)
    slope_24 = compute_linear_regression_slope(close, 24)
    df["slope_acceleration_1h_12_24"] = safe_ratio(slope_12 - slope_24, atr_14)
    df["ema_slope_acceleration_1h"] = ema_fast_slow - ema_fast_slow.shift(3)
    df["volatility_acceleration_1h"] = atr_ratio_1h - atr_ratio_1h.shift(3)

    hour = df["timestamp"].dt.hour
    df["hour_sin_1h"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos_1h"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["is_weekend_1h"] = (df["timestamp"].dt.dayofweek >= 5).astype(float)
    return df


def add_features(df):
    """Ð¢Ð¾Ð»ÑŒÐºÐ¾ 1H Ñ„Ð¸Ñ‡Ð¸ Ð¸Ð· Ñ‚ÐµÐºÑƒÑ‰ÐµÐ³Ð¾ Ð½Ð°Ð±Ð¾Ñ€Ð°."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    candle_range = (high - low).replace(0, np.nan)

    for period in (6, 12, 24):
        df[f"return_1h_{period}"] = np.log(close / close.shift(period))

    log_return_1h_1 = np.log(close / close.shift(1))
    df["realized_vol_1h"] = log_return_1h_1.rolling(REALIZED_VOL_WINDOW_1H).std()

    ema_fast = close.ewm(span=EMA_FAST_WINDOW, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW_WINDOW, adjust=False).mean()
    df["ema_fast_slow"] = safe_ratio(ema_fast - ema_slow, ema_slow)

    atr_14 = compute_atr(df["high"], df["low"], close, length=14)
    atr_100 = compute_atr(df["high"], df["low"], close, length=100)
    atr_6 = compute_atr(df["high"], df["low"], close, length=6)
    atr_48 = compute_atr(df["high"], df["low"], close, length=48)
    df["atr_ratio_1h"] = safe_ratio(atr_14, atr_100)
    df["volatility_regime_change_1h"] = safe_ratio(atr_6, atr_48)

    # beta disabled: signed-volume imbalance over the last 12 hourly candles
    # signed_volume_1h = volume * np.sign(close - open_)
    # df["volume_imbalance_1h_12"] = safe_ratio(
    #     signed_volume_1h.rolling(12).sum(),
    #     volume.rolling(12).sum(),
    # )
    # beta disabled: Parkinson-to-realized volatility ratio for intrabar range information
    # parkinson_component_1h = np.log(safe_ratio(high, low)) ** 2
    # parkinson_vol_1h = np.sqrt(
    #     parkinson_component_1h.rolling(REALIZED_VOL_WINDOW_1H).mean() / (4.0 * np.log(2.0))
    # )
    # df["parkinson_ratio"] = safe_ratio(parkinson_vol_1h, df["realized_vol_1h"])
    # beta disabled: continuous choppiness index, intended to complement adx_4h
    # prev_close = close.shift(1)
    # true_range_1h = pd.concat(
    #     [
    #         high - low,
    #         (high - prev_close).abs(),
    #         (low - prev_close).abs(),
    #     ],
    #     axis=1,
    # ).max(axis=1)
    # choppiness_range_1h = high.rolling(CHOPPINESS_WINDOW_1H).max() - low.rolling(CHOPPINESS_WINDOW_1H).min()
    # choppiness_ratio_1h = safe_ratio(
    #     true_range_1h.rolling(CHOPPINESS_WINDOW_1H).sum(),
    #     choppiness_range_1h,
    # )
    # df["choppiness_index"] = 100.0 * np.log10(choppiness_ratio_1h) / np.log10(CHOPPINESS_WINDOW_1H)

    low_24 = df["low"].rolling(24).min()
    high_24 = df["high"].rolling(24).max()
    df["price_position_1h"] = safe_ratio(close - low_24, high_24 - low_24)
    df["distance_to_support_1h"] = safe_ratio(close - low_24, atr_14)
    df["distance_to_resistance_1h"] = safe_ratio(high_24 - close, atr_14)
    if bool(getattr(cfg, "ENABLE_TEST_FEATURES", False)):
        df = add_test_features(df, atr_14, df["ema_fast_slow"], df["atr_ratio_1h"])
    df["linear_regression_slope_atr_1h_12"] = safe_ratio(compute_linear_regression_slope(close, 12), atr_14)
    df["linear_regression_slope_atr_1h_24"] = safe_ratio(compute_linear_regression_slope(close, 24), atr_14)
    df["volume_mean_3_1h"] = volume.rolling(3).mean()
    if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", True)):
        df["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(close, atr_14, df["realized_vol_1h"])
        df["barrier_take_pct"] = compute_dynamic_barrier_take_pct(df["barrier_stop_pct"])
    else:
        df["barrier_stop_pct"] = float(SL_PCT)
        df["barrier_take_pct"] = float(TP_PCT)
    return df


def build_htf_feature_frame(htf_df, symbol):
    """Ð¤Ð¸Ñ‡Ð¸ 4H ÑÐ¾ ÑÐ´Ð²Ð¸Ð³Ð¾Ð¼ Ð½Ð° Ð¾Ð´Ð½Ñƒ ÑÐ²ÐµÑ‡Ñƒ, Ñ‡Ñ‚Ð¾Ð±Ñ‹ Ð½Ðµ ÑÐ¼Ð¾Ñ‚Ñ€ÐµÑ‚ÑŒ Ð² Ð½ÐµÐ·Ð°ÐºÑ€Ñ‹Ñ‚Ñ‹Ð¹ HTF-Ð±Ð°Ñ€."""
    htf = htf_df.copy().sort_values("timestamp").reset_index(drop=True)
    close = htf["close"]
    high = htf["high"]
    low = htf["low"]

    for period in (1, 3, 7, 14):
        htf[f"return_4h_{period}"] = np.log(close / close.shift(period))

    log_return_4h_1 = np.log(close / close.shift(1))
    htf["realized_vol_4h_returns_20"] = log_return_4h_1.rolling(REALIZED_VOL_WINDOW_4H).std()
    htf["adx_4h"] = compute_adx(high, low, close, length=14)

    low_14 = low.rolling(RANGE_WINDOW_4H).min()
    high_14 = high.rolling(RANGE_WINDOW_4H).max()
    atr_14 = compute_atr(high, low, close, length=14)
    htf["price_position_4h"] = safe_ratio(close - low_14, high_14 - low_14)
    htf["distance_to_rolling_high_4h"] = safe_ratio(close - high_14, atr_14)
    htf["distance_to_rolling_low_4h"] = safe_ratio(close - low_14, atr_14)
    ema_base_4h = close.ewm(span=EMA_SLOPE_BASE_WINDOW_4H, adjust=False).mean()
    htf["ema_slope_4h"] = safe_ratio(compute_linear_regression_slope(ema_base_4h, EMA_SLOPE_WINDOW_4H), atr_14)
    rolling_vwap_4h = compute_rolling_vwap(close, high, low, htf["volume"], VWAP_WINDOW_4H)
    vwap_distance_4h = close - rolling_vwap_4h
    vwap_distance_mean_4h = vwap_distance_4h.rolling(VWAP_WINDOW_4H).mean()
    vwap_distance_std_4h = vwap_distance_4h.rolling(VWAP_WINDOW_4H).std().replace(0, np.nan)
    htf["zscore_vs_vwap_4h"] = (vwap_distance_4h - vwap_distance_mean_4h) / vwap_distance_std_4h
    htf["volume_mean_3_4h_per_hour"] = htf["volume"].rolling(3).mean() / 4.0
    htf["symbol"] = symbol

    htf_feature_columns = [
        "return_4h_1",
        "return_4h_3",
        "return_4h_7",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "zscore_vs_vwap_4h",
        "price_position_4h",
        "adx_4h",
        "distance_to_rolling_high_4h",
        "distance_to_rolling_low_4h",
        "volume_mean_3_4h_per_hour",
    ]
    htf[htf_feature_columns] = htf[htf_feature_columns].shift(1)
    return htf[["timestamp", "symbol"] + htf_feature_columns]


def build_cross_sectional_feature_map(feature_map, source_column, output_column):
    rank_frames = []
    for symbol, source_df in feature_map.items():
        if source_column not in source_df.columns:
            continue
        frame = source_df[["timestamp", source_column]].copy()
        frame = frame.dropna(subset=[source_column])
        if frame.empty:
            continue
        frame["symbol"] = symbol
        rank_frames.append(frame)

    if not rank_frames:
        return {}

    rank_df = pd.concat(rank_frames, ignore_index=True)
    rank_df[output_column] = rank_df.groupby("timestamp")[source_column].transform(normalize_rank)

    feature_map_by_symbol = {}
    for symbol in feature_map:
        symbol_rank_df = rank_df.loc[rank_df["symbol"] == symbol, ["timestamp", output_column]].copy()
        feature_map_by_symbol[symbol] = symbol_rank_df
    return feature_map_by_symbol


def merge_feature_map(feature_map, extra_feature_map, output_columns):
    if isinstance(output_columns, str):
        output_columns = [output_columns]

    enriched = {}
    for symbol, source_df in feature_map.items():
        extra_df = extra_feature_map.get(symbol)
        merged = source_df.copy()
        if extra_df is not None and not extra_df.empty:
            merged = merged.merge(extra_df, on="timestamp", how="left")
        else:
            for column in output_columns:
                merged[column] = np.nan
        enriched[symbol] = merged
    return enriched


def build_shared_market_context_frame(feature_map, source_column, context_builders):
    context_frames = []
    for source_df in feature_map.values():
        if source_column not in source_df.columns:
            continue
        frame = source_df[["timestamp", source_column]].copy()
        frame = frame.dropna(subset=[source_column])
        if not frame.empty:
            context_frames.append(frame)

    if not context_frames:
        return pd.DataFrame(columns=["timestamp"] + list(context_builders.keys()))

    context_source = pd.concat(context_frames, ignore_index=True)
    grouped = context_source.groupby("timestamp")[source_column]
    context_df = pd.DataFrame({"timestamp": grouped.size().index})
    for output_column, builder in context_builders.items():
        context_df[output_column] = grouped.apply(builder).values
    return context_df


def add_shared_market_context(feature_map, context_df, output_columns):
    enriched = {}
    for symbol, source_df in feature_map.items():
        merged = source_df.copy()
        if context_df is not None and not context_df.empty:
            merged = merged.merge(context_df, on="timestamp", how="left")
        else:
            for column in output_columns:
                merged[column] = np.nan
        enriched[symbol] = merged
    return enriched


def build_cross_sectional_rank(htf_feature_map):
    return build_cross_sectional_feature_map(
        htf_feature_map,
        source_column="return_4h_3",
        output_column="cross_sectional_rank_4h",
    )


def add_cross_sectional_rank(htf_feature_map):
    rank_map = build_cross_sectional_rank(htf_feature_map)
    return merge_feature_map(htf_feature_map, rank_map, "cross_sectional_rank_4h")


def add_btc_relative_feature_block(base_feature_map):
    btc_df = base_feature_map.get("BTC/USDT")
    if btc_df is None or btc_df.empty:
        logger.warning("BTC/USDT base feature frame is unavailable, BTC-relative features will be NaN")
        enriched = {}
        for symbol, df in base_feature_map.items():
            merged = df.copy()
            merged["relative_strength_vs_btc_24h"] = np.nan
            merged["beta_to_btc_24h"] = np.nan
            merged["residual_return_24h"] = np.nan
            enriched[symbol] = merged
        return enriched

    btc_reference = btc_df[["timestamp", "close", "return_1h_6", "return_1h_24"]].copy().rename(
        columns={
            "close": "btc_close",
            "return_1h_6": "btc_return_1h_6",
            "return_1h_24": "btc_return_1h_24",
        }
    )

    enriched = {}
    for symbol, df in base_feature_map.items():
        merged = df.copy().merge(btc_reference, on="timestamp", how="left")
        asset_return_1h = np.log(merged["close"] / merged["close"].shift(1))
        btc_return_1h = np.log(merged["btc_close"] / merged["btc_close"].shift(1))
        btc_var_24h = btc_return_1h.rolling(24).var().replace(0, np.nan)
        beta_24h = asset_return_1h.rolling(24).cov(btc_return_1h)
        beta_24h = safe_ratio(beta_24h, btc_var_24h)

        merged["relative_strength_vs_btc_24h"] = merged["return_1h_24"] - merged["btc_return_1h_24"]
        merged["beta_to_btc_24h"] = beta_24h
        merged["residual_return_24h"] = merged["return_1h_24"] - (beta_24h * merged["btc_return_1h_24"])

        if symbol == "BTC/USDT":
            merged["relative_strength_vs_btc_24h"] = 0.0
            merged["beta_to_btc_24h"] = 1.0
            merged["residual_return_24h"] = 0.0

        merged.drop(columns=["btc_close", "btc_return_1h_6", "btc_return_1h_24"], inplace=True)
        enriched[symbol] = merged
    return enriched


def build_test_feature_context(base_feature_map, htf_feature_map):
    base_feature_map = add_btc_relative_feature_block(base_feature_map)
    base_feature_map = merge_feature_map(
        base_feature_map,
        build_cross_sectional_feature_map(
            base_feature_map,
            source_column="ema_fast_slow",
            output_column="cross_sectional_rank_ema_fast_slow_1h",
        ),
        "cross_sectional_rank_ema_fast_slow_1h",
    )

    market_breadth_1h = build_shared_market_context_frame(
        base_feature_map,
        source_column="ema_fast_slow",
        context_builders={
            "market_breadth_ema_fast_slow_1h": lambda series: float((series > 0).mean()),
        },
    )
    base_feature_map = add_shared_market_context(
        base_feature_map,
        market_breadth_1h,
        ["market_breadth_ema_fast_slow_1h"],
    )

    market_context_4h = build_shared_market_context_frame(
        htf_feature_map,
        source_column="return_4h_3",
        context_builders={
            "market_breadth_pos_return_4h_3": lambda series: float((series > 0).mean()),
            "market_dispersion_return_4h_3": lambda series: float(np.nanstd(series.to_numpy(dtype=float), ddof=0)),
        },
    )
    htf_feature_map = add_shared_market_context(
        htf_feature_map,
        market_context_4h,
        ["market_breadth_pos_return_4h_3", "market_dispersion_return_4h_3"],
    )
    return base_feature_map, htf_feature_map


def add_htf_features(df, htf_df):
    """
    Ð”Ð¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¸Ðµ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð·Ð°Ð´Ð°Ð½Ð½Ñ‹Ñ… 4H Ñ„Ð¸Ñ‡ÐµÐ¹.
    HTF-Ñ„Ñ€ÐµÐ¹Ð¼ ÑƒÐ¶Ðµ Ð¿Ð¾Ð´Ð³Ð¾Ñ‚Ð¾Ð²Ð»ÐµÐ½ ÑÐ¾ shift(1), Ñ‡Ñ‚Ð¾Ð±Ñ‹ Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ Ð½ÐµÐ·Ð°ÐºÑ€Ñ‹Ñ‚ÑƒÑŽ 4H ÑÐ²ÐµÑ‡Ñƒ.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    htf = htf_df.copy().sort_values("timestamp").reset_index(drop=True)
    raw_ohlcv_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    if "return_4h_1" not in htf.columns and raw_ohlcv_columns.issubset(htf.columns):
        symbol = htf["symbol"].iloc[0] if "symbol" in htf.columns and not htf["symbol"].empty else "unknown"
        htf = build_htf_feature_frame(htf, symbol=symbol)

    htf_merge_columns = [
        "timestamp",
        "return_4h_1",
        "return_4h_3",
        "return_4h_7",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "zscore_vs_vwap_4h",
        "price_position_4h",
        "adx_4h",
        "distance_to_rolling_high_4h",
        "distance_to_rolling_low_4h",
        "volume_mean_3_4h_per_hour",
        "cross_sectional_rank_4h",
        "market_breadth_pos_return_4h_3",
        "market_dispersion_return_4h_3",
    ]
    available_htf_columns = [column for column in htf_merge_columns if column in htf.columns]
    df = pd.merge_asof(
        df,
        htf[available_htf_columns],
        on="timestamp",
        direction="backward",
    )
    for required_column in htf_merge_columns:
        if required_column not in df.columns:
            df[required_column] = np.nan
    realized_vol_4h_per_hour = df["realized_vol_4h_returns_20"] / np.sqrt(4.0)
    df["vol_ratio"] = safe_ratio(df["realized_vol_1h"], realized_vol_4h_per_hour)
    df.drop(columns=["volume_mean_3_1h", "volume_mean_3_4h_per_hour"], inplace=True)
    return df


def add_post_merge_test_features(df):
    df = df.copy()

    breadth = df["market_breadth_ema_fast_slow_1h"]
    dispersion = df["market_dispersion_return_4h_3"]
    breadth_mean = breadth.rolling(MARKET_ZSCORE_WINDOW).mean()
    breadth_std = breadth.rolling(MARKET_ZSCORE_WINDOW).std().replace(0, np.nan)

    df["delta_market_breadth_ema_fast_slow_1h"] = breadth.diff(1)
    df["market_breadth_ema_fast_slow_1h_zscore"] = (breadth - breadth_mean) / breadth_std
    df["ema_fast_slow_x_market_breadth_ema_fast_slow_1h"] = df["ema_fast_slow"] * breadth
    df["trend_efficiency_24h_x_volatility_regime_change_1h"] = (
        df["trend_efficiency_24h"] * df["volatility_regime_change_1h"]
    )
    return df


def compute_clean_pnl(direction, entry_price, exit_price):
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    return raw_pnl - (TAKER_COM + TAKER_COM)


def resolve_trade_exit(direction, entry_price, next_open, next_high, next_low, stop_pct, take_pct):
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)

        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - SLIPPAGE)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)

        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + SLIPPAGE)
            return exit_price, "TP"

    return None, None


def simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, start_idx, direction):
    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 + SLIPPAGE) if direction == 1 else base_open * (1 - SLIPPAGE)
    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if np.isnan(stop_pct) or np.isnan(take_pct):
        return 0.0, None

    for j in range(1, HORIZON + 1):
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


def triple_barrier_labeling(df):
    """Ð Ð°Ð·Ð¼ÐµÑ‚ÐºÐ° Ð´Ð°Ð½Ð½Ñ‹Ñ… (Teacher) â€” dynamic ATR/realized-vol stop Ð¸ TP Ð¾Ñ‚ RR.
    Ð•ÑÐ»Ð¸ Ð²Ð½ÑƒÑ‚Ñ€Ð¸ Ð¾Ð´Ð½Ð¾Ð¹ ÑÐ²ÐµÑ‡Ð¸ Ð·Ð°Ð´ÐµÑ‚Ñ‹ Ð¾Ð±Ð° Ð±Ð°Ñ€ÑŒÐµÑ€Ð°, Ð¿Ñ€Ð¸Ð¾Ñ€Ð¸Ñ‚ÐµÑ‚ Ð²ÑÐµÐ³Ð´Ð° Ñƒ SL."""
    labels = []

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - HORIZON):
        label = 0
        long_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=1)
        short_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=-1)

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1

        labels.append(label)

    labels.extend([0] * HORIZON)
    df['Target'] = labels
    return df



def finalize_feature_frame(df):
    df = df.copy()
    if HORIZON > 0:
        if len(df) <= HORIZON:
            return df.iloc[0:0][OUTPUT_COLUMNS].copy()
        df = df.iloc[:-HORIZON].copy()

    df = df[OUTPUT_COLUMNS].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_exchange_service() -> ExchangeContract:
    exchange_name = str(getattr(cfg, "ACTIVE_EXCHANGE", "bybit")).strip().lower()
    if exchange_name == "bybit":
        return BybitService()
    if exchange_name == "binance":
        return BinanceService()
    raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {exchange_name}")


def main():
    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()

    symbols_to_load = [exchange_service.normalize_symbol(symbol) for symbol in dict.fromkeys(SYMBOLS)]
    symbol_names = [str(symbol) for symbol in symbols_to_load]

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        logger.info(f"Loading {symbol_name} {TIMEFRAME} from {START_DATE}...")
        loaded = repository.sync_candles(exchange_service, symbol, TIMEFRAME, START_DATE, END_DATE)
        logger.info(f"{symbol_name} {TIMEFRAME}: {loaded} new candles")

        logger.info(f"Loading {symbol_name} {HTF_TIMEFRAME} from {START_DATE}...")
        htf_loaded = repository.sync_candles(exchange_service, symbol, HTF_TIMEFRAME, START_DATE, END_DATE)
        logger.info(f"{symbol_name} {HTF_TIMEFRAME}: {htf_loaded} new candles")

    base_1h_map = {}
    htf_feature_map = {}
    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = repository.load_candles(symbol, TIMEFRAME)
        htf_df = repository.load_candles(symbol, HTF_TIMEFRAME)
        if df.empty or htf_df.empty:
            logger.warning(f"{symbol_name}: no data in DB (1h={len(df)}, 4h={len(htf_df)})")
            continue

        logger.info(f"{symbol_name}: 1h={len(df)}, 4h={len(htf_df)} rows")
        base_1h_map[symbol_name] = add_features(df)
        htf_feature_map[symbol_name] = build_htf_feature_frame(htf_df, symbol_name)

    rank_source_map = {symbol: htf_feature_map[symbol] for symbol in symbol_names if symbol in htf_feature_map}
    ranked_map = add_cross_sectional_rank(rank_source_map)
    for symbol, ranked_htf in ranked_map.items():
        htf_feature_map[symbol] = ranked_htf
    for symbol in htf_feature_map:
        if symbol not in ranked_map:
            htf_feature_map[symbol]["cross_sectional_rank_4h"] = np.nan

    if bool(getattr(cfg, "ENABLE_TEST_FEATURES", False)):
        base_test_source_map = {symbol: base_1h_map[symbol] for symbol in symbol_names if symbol in base_1h_map}
        htf_test_source_map = {symbol: htf_feature_map[symbol] for symbol in symbol_names if symbol in htf_feature_map}
        enriched_base_map, enriched_htf_map = build_test_feature_context(base_test_source_map, htf_test_source_map)
        for symbol, enriched_base in enriched_base_map.items():
            base_1h_map[symbol] = enriched_base
        for symbol, enriched_htf in enriched_htf_map.items():
            htf_feature_map[symbol] = enriched_htf

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = base_1h_map.get(symbol_name)
        htf_df = htf_feature_map.get(symbol_name)
        if df is None or htf_df is None:
            logger.warning(f"{symbol_name}: skipped, missing prepared feature inputs")
            continue

        df = add_htf_features(df, htf_df)
        if bool(getattr(cfg, "ENABLE_TEST_FEATURES", False)):
            df = add_post_merge_test_features(df)
        df = triple_barrier_labeling(df)
        df = finalize_feature_frame(df)
        repository.save_features(symbol, df)
        logger.info(f"{symbol_name}: saved {len(df)} rows with the requested feature set")


if __name__ == '__main__':
    main()
