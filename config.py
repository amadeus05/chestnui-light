from pathlib import Path
# --- BASE ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "market_data.db")
SYMBOLS = [
    "BTC/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "SOL/USDT",

    "XRP/USDT",
    "XLM/USDT",
    "ADA/USDT",
    "TRX/USDT",
    "XMR/USDT",

    "ATOM/USDT",
    "DOGE/USDT",
    
    # "LINK/USDT",
    # "NEAR/USDT",
    # "RENDER/USDT",
    # "ARB/USDT",
    # "HBAR/USDT",
    # "MATIC/USDT",
    # "OP/USDT",
    # "TIA/USDT",
    # "FET/USDT",
    # "SEI/USDT",
    # "WLD/USDT",
    # "INJ/USDT",

    # "AVAX/USDT",
    # "SUI/USDT",
    # "STX/USDT",


    # "TON/USDT",
    # "APT/USDT",
    # "TAO/USDT",
]

TIMEFRAME = "5m"
HTF_TIMEFRAME = "1h"  # Старший таймфрейм для мульти-TF фичей

# --- DATA LOADING ---
ACTIVE_EXCHANGE = "bybit"  # bybit: bybit, binance

START_DATE = "2023-01-01"
END_DATE = "2026-03-27 21:00:00"

BACKTEST_INITIAL_BALANCE = 100
ENABLE_PROD_TRAINING = False
MODEL_NAME = "lightgbm_long_only"

ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = True
MANUAL_DISABLED_FEATURE_COLUMNS = []

# --- FEATURE BUILD ---
DONCHIAN_LENGTH = 96
LWTI_PERIOD = 25
LWTI_SMOOTHING_PERIOD = 20
VOLUME_MA_LENGTH = 30
HTF_SR_LOOKBACK = 24
HTF_SR_BUFFER_ATR = 0.5
CANDLE_STRUCTURE_ATR_LENGTH = 14
CANDLE_STRUCTURE_EFFICIENCY_WINDOW = 12
MARKET_CONTEXT_CORR_WINDOW = 96
MARKET_CONTEXT_ATR_LENGTH = 14
COMPRESSION_WIDTH_ZSCORE_LOOKBACK = 288
COMPRESSION_SHORT_VOL_WINDOW = 12
COMPRESSION_LONG_VOL_WINDOW = 96
COMPRESSION_CHANGE_BARS = 3
COMPRESSION_RANGE_FAST_WINDOW = 3
COMPRESSION_RANGE_SLOW_WINDOW = 12
VOLUME_PRESSURE_WINDOW = 12

FEATURE_PROFILES = {
    "empty": [],
    "donchian_lwti": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
    ],
    "donchian_lwti_compression": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
        "range_expansion_ratio_3_12",
        "range_expansion_change_3",
        "volume_ratio_change_3",
    ],
    "donchian_lwti_compression_lite": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
    ],
    "donchian_lwti_compression_lite_volume_v1_lite": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
        "up_volume_share_12",
        "net_candle_volume_bias_12",
        "effort_vs_result_12",
    ],
    "donchian_lwti_compression_lite_volume_v1_lite_candle_v1": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
        "up_volume_share_12",
        "net_candle_volume_bias_12",
        "effort_vs_result_12",
        "close_location_in_bar",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "true_range_to_atr_14",
        "efficiency_ratio_12",
    ],
    "donchian_lwti_compression_lite_volume_v1_lite_candle_v1_btc_relative_v1": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
        "up_volume_share_12",
        "net_candle_volume_bias_12",
        "effort_vs_result_12",
        "close_location_in_bar",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "true_range_to_atr_14",
        "efficiency_ratio_12",
        "relative_return_vs_btc_4h",
        "relative_return_vs_btc_3d",
        "beta_to_btc_96",
    ],
    "donchian_lwti_compression_lite_volume_v1_lite_candle_v1_lite": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "donchian_width_zscore_96_288",
        "donchian_width_change_3",
        "donchian_width_acceleration_3",
        "realized_vol_ratio_12_96",
        "realized_vol_change_3",
        "up_volume_share_12",
        "net_candle_volume_bias_12",
        "effort_vs_result_12",
        "close_location_in_bar",
        "true_range_to_atr_14",
        "efficiency_ratio_12",
    ],
    "donchian_lwti_context": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
        "htf_return_1",
        "htf_return_3",
        "htf_ema_fast_slow",
        "htf_ema_slope",
        "htf_adx_14",
        "htf_realized_vol_20",
        "htf_price_position_24",
        "htf_trend_efficiency_24",
        "btc_return_24h",
        "relative_strength_vs_btc_24h",
        "intraday_time_sin",
        "intraday_time_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_asia_session",
        "is_london_session",
        "is_ny_session",
        "is_london_ny_overlap",
    ],
    "v1": [
        "donchian_width_pct_96",
        "donchian_mid_distance_atr_96",
        "donchian_close_position_96",
        "donchian_upper_break_atr_96",
        "donchian_lower_break_atr_96",
        "donchian_upper_touch_96",
        "donchian_lower_touch_96",
        "lwti_25_20",
        "lwti_centered_25_20",
        "lwti_slope_3",
        "lwti_long_bias_25_20",
        "volume_ratio_30",
        "volume_green_1",
        "volume_red_1",
        "volume_green_above_ma_30",
        "volume_red_above_ma_30",
        "htf_resistance_distance_atr_24",
        "htf_support_distance_atr_24",
        "htf_near_resistance_24",
        "htf_near_support_24",
        "breakout_long_score",
        "breakout_short_score",
        "breakout_long_signal",
        "breakout_short_signal",
    ],
}
FEATURE_BUILD_REQUEST = {
    "profile": "donchian_lwti_compression_lite_volume_v1_lite_candle_v1_btc_relative_v1",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

# --- ML LABELING (Triple Barrier) ---
HORIZON = 12
TP_PCT = 0.03   # legacy fixed TP, kept for backward compatibility
SL_PCT = 0.015  # legacy fixed SL, kept for backward compatibility
TIME_EXIT_NEUTRAL_BAND = 0.01
# Значимое движение: gross_return > average_spread + 2*TAKER_COM
# (в etl long_pnl уже net = gross - 2*TAKER_COM → класс 1 при long_pnl > average_spread)
LABEL_USE_SIGNIFICANT_RETURN = True
LABEL_AVERAGE_SPREAD_PCT = 0.0  # нижняя граница / фикс. спред, если rolling выключен
LABEL_USE_ROLLING_SPREAD_PROXY = True
LABEL_SPREAD_ROLLING_BARS = 24

# --- LightGBM (train.py) ---
LGBM_N_ESTIMATORS = 2000
LGBM_LEARNING_RATE = 0.01
LGBM_NUM_LEAVES = 31  # при необходимости попробуйте 48
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_LOG_EVAL_PERIOD = 50

# --- TRAIN VALIDATION ---
TRAIN_VALIDATION_MODE = "walk_forward"  # "single" | "walk_forward"
WF_TRAIN_MONTHS = 6
WF_VAL_MONTHS = 1
WF_TEST_MONTHS = 1
WF_STEP_MONTHS = 1
WF_EMBARGO_BARS = HORIZON

# --- BARRIERS ---
BARRIER_MODE = "donchian_midline_rr"  # "dynamic" | "fixed" | "donchian_midline_rr"
USE_DYNAMIC_BARRIERS = False  # legacy flag, kept for backward compatibility
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 1.10
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06
STRATEGY_BARRIER_TP_TO_SL_RATIO = 2.0
STRATEGY_BARRIER_LOCAL_EXTREME_LOOKBACK = 12
STRATEGY_BARRIER_MAX_MIDLINE_PCT = 0.03
STRATEGY_BARRIER_MIN_PCT = 0.0

# --- EVENT FILTER (binary side model candidate universe) ---
ENABLE_EVENT_FILTER = False
EVENT_FILTER_SIDE = "both"
EVENT_FILTER_MIN_EMA_FAST_SLOW = 0.003
EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW = EVENT_FILTER_MIN_EMA_FAST_SLOW
EVENT_FILTER_MIN_ADX_4H = 18.0
EVENT_FILTER_MIN_ADX_HTF = EVENT_FILTER_MIN_ADX_4H
EVENT_FILTER_MIN_REALIZED_VOL_1H = 0.003
EVENT_FILTER_MIN_REALIZED_VOL_MAIN = EVENT_FILTER_MIN_REALIZED_VOL_1H
EVENT_FILTER_MAX_REALIZED_VOL_1H = 0.05
EVENT_FILTER_MAX_REALIZED_VOL_MAIN = EVENT_FILTER_MAX_REALIZED_VOL_1H

# --- BACKTEST EVENT FILTER (stricter execution gate) ---
BACKTEST_ENABLE_EVENT_FILTER = False
BACKTEST_EVENT_FILTER_SIDE = "long"
BACKTEST_EVENT_FILTER_MIN_EMA_FAST_SLOW = 0.004
BACKTEST_EVENT_FILTER_MIN_ADX_4H = 22.0
BACKTEST_EVENT_FILTER_MIN_REALIZED_VOL_1H = 0.004
BACKTEST_EVENT_FILTER_MAX_REALIZED_VOL_1H = 0.03

# --- RAW REBUILD SAFETY ---
ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False

# --- TRADING ---
TAKER_COM = 0.0004
MAKER_COM = 0.0002
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
LONG_PROBA_THRESHOLD = 0.50
DIRECTIONAL_PROBA_THRESHOLD = LONG_PROBA_THRESHOLD
CONFIDENCE_THRESHOLD = LONG_PROBA_THRESHOLD
MIN_SIGNAL_GAP = 0.01
ALLOW_LONGS = True
ALLOW_SHORTS = False
BACKTEST_REALTIME_FEATURES = False
BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 1     # 1 = берем лучший сигнал на баре, >1 = топ-N сигналов
BACKTEST_MAX_OPEN_POSITIONS = 3            # максимум одновременно открытых позиций
BACKTEST_SL_COOLDOWN_BARS = 12
BACKTEST_MAX_SL_PER_DAY = 3
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = 2
BACKTEST_REDUCED_RISK_PER_TRADE = 0.005

# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_CHARTS_DIR = Path("backtest_charts")
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)
