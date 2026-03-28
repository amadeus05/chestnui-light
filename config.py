from pathlib import Path

# --- ОСНОВНЫЕ ---
DB_PATH = "market_data.db"
SYMBOLS = [
    "BTC/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "XMR/USDT",
    "LINK/USDT",
    "NEAR/USDT",
    "RENDER/USDT",
    "TRX/USDT",
    "ADA/USDT",
    "ATOM/USDT",
    "ARB/USDT",
    "AVAX/USDT",
    "XLM/USDT",
    "HBAR/USDT",
    "MATIC/USDT",
    "OP/USDT",
    "TIA/USDT",
    "FET/USDT",
    "SEI/USDT",
    "WLD/USDT",
    "SUI/USDT",
    "STX/USDT",
    "INJ/USDT",


    # "DOGE/USDT",

    # "TON/USDT",
    # "APT/USDT",
    # # "TAO/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"  # Старший таймфрейм для мульти-TF фичей

# --- DATA LOADING ---
START_DATE = "2023-01-01"  # Дата начала загрузки данных
END_DATE = "2026-03-27 21:00:00"  # Fixed cutoff for reproducible ETL/train/bt runs
BYBIT_LIMIT = 1000         # Лимит свечей Bybit за один запрос для kline
BYBIT_RETRY_SLEEP = 0.33    # Базовая пауза/бекофф между повторами запросов
BACKTEST_INITIAL_BALANCE = 100
ENABLE_PROD_TRAINING = False     # False = не трогать последние 15% (для честного теста), True = учить на всём (перед запуском)

ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = False

# --- ML LABELING (Triple Barrier) ---
HORIZON = 16
TP_PCT = 0.03   # legacy fixed TP, kept for backward compatibility
SL_PCT = 0.015   # legacy fixed SL, kept for backward compatibility

# --- DYNAMIC BARRIERS ---
USE_DYNAMIC_BARRIERS = True
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06

# --- TRADING ---
TAKER_COM = 0.0004
MAKER_COM = 0.0002
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
DIRECTIONAL_PROBA_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = DIRECTIONAL_PROBA_THRESHOLD
MIN_SIGNAL_GAP = 0.10
ALLOW_LONGS = True
ALLOW_SHORTS = True
BACKTEST_REALTIME_FEATURES = False
BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 10     # 1 = берем лучший сигнал на баре, >1 = топ-N сигналов
BACKTEST_MAX_OPEN_POSITIONS = 10            # максимум одновременно открытых позиций
BACKTEST_SAVE_TRADE_CHARTS = False          # сохранять HTML-график по каждой закрытой сделке
ENABLE_TEST_FEATURES = True   # Включать ли test features в трейдинг
# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_CHARTS_DIR = Path("backtest_charts")
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)
