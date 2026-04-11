import os
from pathlib import Path


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


# --- BASE ---
DB_PATH = str(_env_str("DB_PATH", "market_data.db"))
SYMBOLS = [
    "BTC/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "SOL/USDT",

    "XRP/USDT",
    "XLM/USDT",
    "ADA/USDT",
    "TRX/USDT",
    # "XMR/USDT",

    # "LINK/USDT",
    # "NEAR/USDT",
    # "RENDER/USDT",
    # "ATOM/USDT",
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


    "DOGE/USDT",
    # "TON/USDT",
    # "APT/USDT",
    # "TAO/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"  # Старший таймфрейм для мульти-TF фичей

# --- DATA LOADING ---
ACTIVE_EXCHANGE = "bybit"  # bybit: bybit, binance
START_DATE = "2023-01-01"
END_DATE = "2026-03-27 21:00:00"
BYBIT_LIMIT = 1000
BYBIT_RETRY_SLEEP = 0.33
BINANCE_BASE_URL = "https://fapi.binance.com"
BINANCE_LIMIT = 1000
BINANCE_TIMEOUT = 20
BINANCE_MAX_WORKERS = 3
BINANCE_RETRY_COUNT = 5
BINANCE_RETRY_SLEEP = 0.25
BINANCE_REQUEST_WEIGHT_LIMIT_PER_MINUTE = 2400
BACKTEST_INITIAL_BALANCE = 100
ENABLE_PROD_TRAINING = False

ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = False
ACTIVE_EXPERIMENT = "hybrid_v1_labels_v2_train"
LABELING_PROFILE = "v1"
TRAINING_PROFILE = "v2"
MANUAL_DISABLED_FEATURE_COLUMNS = [
    "return_1h_24",
    "return_4h_3",
    # Низкий importance (< 1000 gain), создают шум:
    "distance_to_session_low_1h",
    "slope_acceleration_1h_12_24",
    "price_position_4h",
    "price_position_1h",
    "distance_to_support_1h",
    "range_compression_1h",
    "hour_cos_1h",
    "distance_to_resistance_1h",
    "relative_strength_vs_btc_24h",
    "return_4h_1",
    "distance_to_session_high_1h",
    "cross_sectional_rank_ema_fast_slow_1h",
    "return_1h_12",
    "return_1h_6",
    "cross_sectional_rank_4h",
    "is_weekend_1h",
    "crowded_longs_score_1h",
    "crowded_shorts_score_1h",
    "premium_index_change_24h",
    "linear_regression_slope_atr_1h_24",
    "distance_to_rolling_low_4h",
    "residual_return_24h",
    # Regime features - оставляем только работающие (см. feature importance)
    # "volatility_regime_change_1h",  # ВКЛЮЧЕН - gain=282, хорошо работает
    # "volatility_acceleration_1h",  # ВКЛЮЧЕН - gain=100
    # "volatility_regime_stability",  # ВКЛЮЧЕН - gain=264, хорошо
    # "vol_of_vol_1h",  # ВКЛЮЧЕН - gain=957, ТОП-6, отлично!
    # "realized_vol_vs_ema",  # ВКЛЮЧЕН - gain=107
    # "trend_persistence_score_24",   # ВКЛЮЧЕН - gain=26, слабый но оставим
    # "trend_efficiency_24h",       # ВКЛЮЧЕН - gain=23, слабый но оставим
    # "trend_persistence_score_12", # ВКЛЮЧЕН - gain=206
    # Volume features - ВКЛЮЧЕНЫ
    # "volume_ratio_1h",              # ВКЛЮЧЕН - gain=294
    # "volume_zscore_1h",             # ВКЛЮЧЕН - gain=206
    # "dollar_volume_zscore_1h",      # ВКЛЮЧЕН - gain=114
    # Отключаем НЕРАБОТАЮЩИЕ признаки (gain=0 или низкий):
    "delta_market_breadth_ema_fast_slow_1h",  # gain=0 - не работает
    "high_vol_stress_indicator",  # gain=0 - не работает
    "breakout_quality_4h",  # gain=0 - не работает (Donchian разрыв?)
    "counter_market_penalty_1h",  # gain=0 - не работает
    "signal_x_high_vol_stress",  # gain=0 - не работает
    "vol_regime_classification",  # gain=85 - слабый, отключаем (категориальный нестабильный)
    "trend_efficiency_x_vol_stability",  # gain=64 - слабый, отключаем
    "signal_market_agreement_1h",  # gain=80 - слабый, отключаем
    "breakout_quality_4h_x_volume_ratio_1h",  # gain=8 - очень слабый, отключаем
    # Отключаем слабые по новому тесту (gain < 100):
    "premium_index_zscore_7d",  # gain=55 - очень слабый
    "volume_zscore_1h",  # gain=62 - очень слабый
    "shorts_overheated_1h",  # gain=121 - слабый
    "trend_persistence_score_24",  # gain=146 - слабый
    "volatility_acceleration_1h",  # gain=167 - слабый
    "dollar_volume_zscore_1h",  # gain=167 - слабый
    "premium_index_1h",  # gain=115 - слабый
    # Отключаем regime интеракции, оставляем только базовые regime признаки
    "realized_vol_vs_ema",  # оставляем vol_of_vol, vol_regime_change, vol_stability
    "ema_fast_slow_x_vol_of_vol",  # интеракция - отключаем
    "trend_efficiency_24h_x_volatility_regime_change_1h",  # интеракция - отключаем
    "market_pressure_x_vol_regime",  # интеракция - отключаем
    "trend_efficiency_24h",  # мало влияет в новом режиме
    "trend_persistence_score_12",  # слабый
    "donchian_width_change_4h",  # можно отключить
    "open_interest_zscore_7d",  # слабый
]
# --- FEATURE BUILD ---
FEATURE_PROFILES = {
    "all": "__all__",
    "empty": [],
    "base_only": [
        "realized_vol_1h",
        "ema_fast_slow",
        "return_1h_24",
        "linear_regression_slope_atr_1h_24",
        "trend_efficiency_24h",
        "atr_ratio_1h",
        "range_compression_1h",
        "price_position_1h",
        "relative_strength_vs_btc_24h",
        "residual_return_24h",
        "return_4h_3",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "market_breadth_pos_return_4h_3",
        "market_dispersion_return_4h_3",
        "zscore_vs_vwap_4h",
        "vol_ratio",
        "price_position_4h",
        "adx_4h",
    ],
}
FEATURE_BUILD_REQUEST = {
    "profile": "all",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

# Hybrid setup:
# - ETL labeling stays on v1.
# - Training-side filtering and feature pruning stay on v2.
# --- ML LABELING (Triple Barrier) ---
HORIZON = 12
TP_PCT = 0.03   # legacy fixed TP, kept for backward compatibility
SL_PCT = 0.015  # legacy fixed SL, kept for backward compatibility

# --- DYNAMIC BARRIERS ---
USE_DYNAMIC_BARRIERS = True
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06

# --- EVENT FILTER (binary side model candidate universe) ---
ENABLE_EVENT_FILTER = True

# Базовые пороги (используются при нормальном режиме)
EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW = 0.004  # Снижено для большего coverage
EVENT_FILTER_MIN_ADX_HTF = 20.0  # Снижено для большего coverage
EVENT_FILTER_MIN_REALIZED_VOL_MAIN = 0.005  # Снижен минимум
EVENT_FILTER_MAX_REALIZED_VOL_MAIN = 0.040  # Увеличен максимум

# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE EVENT FILTER - 2026-04-05
# Автоматически подстраивает пороги под текущий режим волатильности
# ═══════════════════════════════════════════════════════════════════
ENABLE_ADAPTIVE_EVENT_FILTER = False  # Отключаем для теста

# Пороги для разных режимов волатильности
ADAPTIVE_FILTER_LOW_VOL = {
    "min_abs_ema_fast_slow": 0.003,  # Меньше тренд нужен при низкой воле
    "min_adx_4h": 18.0,
    "min_realized_vol_1h": 0.004,
    "max_realized_vol_1h": 0.015,
}

ADAPTIVE_FILTER_NORMAL_VOL = {
    "min_abs_ema_fast_slow": 0.004,
    "min_adx_4h": 20.0,
    "min_realized_vol_1h": 0.005,
    "max_realized_vol_1h": 0.040,
}

ADAPTIVE_FILTER_HIGH_VOL = {
    "min_abs_ema_fast_slow": 0.006,  # Больше тренд нужен при высокой воле
    "min_adx_4h": 24.0,
    "min_realized_vol_1h": 0.020,
    "max_realized_vol_1h": 0.080,
}

# Параметры определения режима (персентили волатильности)
ADAPTIVE_VOL_PERCENTILE_LOW = 0.25   # 25-й персентиль = low vol
ADAPTIVE_VOL_PERCENTILE_HIGH = 0.75  # 75-й персентиль = high vol
ADAPTIVE_VOL_LOOKBACK_BARS = 96       # 4 дня для расчета персентилей

# ═══════════════════════════════════════════════════════════════════
# TEMPORAL SAMPLE WEIGHTING - 2026-04-05
# Усиленное взвешивание для адаптации к смене режима
# ═══════════════════════════════════════════════════════════════════
SAMPLE_WEIGHT_HALF_LIFE_DAYS = 90.0   # Было 365 - слишком медленно для крипты
REGIME_AWARE_WEIGHTING = True           # Дополнительный буст свежим данным
REGIME_RECENT_DAYS_BOOST = 30.0         # Сколько дней считать "свежими"
REGIME_RECENT_BOOST_FACTOR = 2.0        # Во сколько раз увеличить вес свежих

# --- RAW REBUILD SAFETY ---
ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False

# --- TRADING ---
TAKER_COM = 0.0004
MAKER_COM = 0.0002
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
DIRECTIONAL_PROBA_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.55
MIN_SIGNAL_GAP = 0.01
ALLOW_LONGS = True
ALLOW_SHORTS = True
BACKTEST_REALTIME_FEATURES = False
BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 1     # 1 = берем лучший сигнал на баре, >1 = топ-N сигналов
BACKTEST_MAX_OPEN_POSITIONS = 1            # максимум одновременно открытых позиций
BACKTEST_SL_COOLDOWN_BARS = 8
BACKTEST_MAX_SL_PER_DAY = 3
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = 2
BACKTEST_REDUCED_RISK_PER_TRADE = 0.005
# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_CHARTS_DIR = Path("backtest_charts")
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)

# --- EXECUTION LOG (paper / live, paper.py) ---
# None → та же БД, что и ETL (DB_PATH). Отдельный файл — только если нужно изолировать WAL.
EXECUTION_DB_PATH = _env_str("EXECUTION_DB_PATH")

# --- EXECUTION DB TYPE ---
# "sqlite" - локальная SQLite база (по умолчанию)
# "supabase" - облачная PostgreSQL через Supabase
EXECUTION_DB_TYPE = str(_env_str("EXECUTION_DB_TYPE", "sqlite")).lower()

# --- SUPABASE CONFIG (только если EXECUTION_DB_TYPE = "supabase") ---
# URL проекта Supabase (например: "https://xxxxxx.supabase.co")
SUPABASE_URL = _env_str("SUPABASE_URL", "https://jjuatlyxubeglxkrpaji.supabase.co")
# SUPABASE_KEY должен быть задан в переменных окружения:
# PowerShell: $env:SUPABASE_KEY="your-anon-key-or-service-key"
SUPABASE_KEY = _env_str("SUPABASE_KEY") or _env_str("SUPABASE_SERVICE_KEY")

PAPER_CLOCK_SYMBOL = "BTC/USDT"
PAPER_MODEL_NAME = "lightgbm_target"
PAPER_INITIAL_BALANCE = BACKTEST_INITIAL_BALANCE
PAPER_MAIN_BARS = 3000
PAPER_HTF_BARS = 900
PAPER_MIN_MAIN_ROWS = 400
PAPER_MIN_HTF_ROWS = 120
PAPER_DAEMON_POLL_SEC = 45.0

