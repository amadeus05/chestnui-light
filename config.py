from pathlib import Path
# --- BASE ---
DB_PATH = "market_data.db"
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

    # "LINK/USDT",
    # "NEAR/USDT",
    # "RENDER/USDT",
    "ATOM/USDT",
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
ACTIVE_EXCHANGE = "binance"  # bybit: bybit, binance
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
MANUAL_DISABLED_FEATURE_COLUMNS = []

# --- FEATURE BUILD ---
FEATURE_PROFILES = {
    "all": "__all__",
    "empty": [],
    "base_only": [
        "return_1h_6",
        "return_1h_12",
        "return_1h_24",
        "realized_vol_1h",
        "ema_fast_slow",
        "linear_regression_slope_atr_1h_12",
        "linear_regression_slope_atr_1h_24",
        "volatility_regime_change_1h",
        "atr_ratio_1h",
        "price_position_1h",
        "return_4h_1",
        "return_4h_3",
        "return_4h_7",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "zscore_vs_vwap_4h",
        "vol_ratio",
        "price_position_4h",
        "adx_4h",
    ],
}
FEATURE_BUILD_REQUEST = {
    "profile": "base_only",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

# --- ML LABELING (Triple Barrier) ---
HORIZON = 16
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
EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW = 0.003
EVENT_FILTER_MIN_ADX_HTF = 18.0
EVENT_FILTER_MIN_REALIZED_VOL_MAIN = 0.003
EVENT_FILTER_MAX_REALIZED_VOL_MAIN = 0.05

# --- RAW REBUILD SAFETY ---
ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False

# --- TRADING ---
TAKER_COM = 0.0004
MAKER_COM = 0.0002
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
DIRECTIONAL_PROBA_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = TRADEABILITY_PROBA_THRESHOLD
MIN_SIGNAL_GAP = 0.01
ALLOW_LONGS = True
ALLOW_SHORTS = True
BACKTEST_REALTIME_FEATURES = False
BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 3     # 1 = берем лучший сигнал на баре, >1 = топ-N сигналов
BACKTEST_MAX_OPEN_POSITIONS = 3            # максимум одновременно открытых позиций
# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_CHARTS_DIR = Path("backtest_charts")
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)
