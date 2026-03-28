import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config as cfg
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from requests.adapters import HTTPAdapter

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KLINE_URL = "https://api.bybit.com/v5/market/kline"
MARK_KLINE_URL = "https://api.bybit.com/v5/market/mark-price-kline"
INDEX_KLINE_URL = "https://api.bybit.com/v5/market/index-price-kline"
PREMIUM_KLINE_URL = "https://api.bybit.com/v5/market/premium-index-price-kline"
OPEN_INTEREST_URL = "https://api.bybit.com/v5/market/open-interest"
FUNDING_HISTORY_URL = "https://api.bybit.com/v5/market/funding/history"
ACCOUNT_RATIO_URL = "https://api.bybit.com/v5/market/account-ratio"
BYBIT_CATEGORY = getattr(cfg, "BYBIT_CATEGORY", "linear")
BYBIT_LIMIT = min(1000, max(1, int(getattr(cfg, "BYBIT_LIMIT", 1000))))
BYBIT_CONTEXT_LIMIT = min(1000, max(1, int(getattr(cfg, "BYBIT_CONTEXT_LIMIT", 1000))))
BYBIT_OI_LIMIT = min(200, max(1, int(getattr(cfg, "BYBIT_OI_LIMIT", 200))))
BYBIT_RATIO_LIMIT = min(500, max(1, int(getattr(cfg, "BYBIT_RATIO_LIMIT", 500))))
BYBIT_FUNDING_LIMIT = min(200, max(1, int(getattr(cfg, "BYBIT_FUNDING_LIMIT", 200))))
BYBIT_TIMEOUT = float(getattr(cfg, "BYBIT_TIMEOUT", 20))
BYBIT_MAX_WORKERS = max(1, int(getattr(cfg, "BYBIT_MAX_WORKERS", 6)))
BYBIT_RETRY_COUNT = max(1, int(getattr(cfg, "BYBIT_RETRY_COUNT", 5)))
BYBIT_RETRY_SLEEP = float(getattr(cfg, "BYBIT_RETRY_SLEEP", 0.3))
DB_WRITE_BATCH_ROWS = max(BYBIT_LIMIT * 10, int(getattr(cfg, "DB_WRITE_BATCH_ROWS", 20_000)))
MARKET_ZSCORE_WINDOW = max(10, int(getattr(cfg, "MARKET_ZSCORE_WINDOW", 96)))
FUNDING_ZSCORE_WINDOW = max(5, int(getattr(cfg, "FUNDING_ZSCORE_WINDOW", 24)))
EMA_FAST_WINDOW = max(2, int(getattr(cfg, "EMA_FAST_WINDOW", 12)))
EMA_SLOW_WINDOW = max(EMA_FAST_WINDOW + 1, int(getattr(cfg, "EMA_SLOW_WINDOW", 48)))
EMA_SLOPE_BASE_WINDOW_4H = max(2, int(getattr(cfg, "EMA_SLOPE_BASE_WINDOW_4H", 21)))
EMA_SLOPE_WINDOW_4H = max(2, int(getattr(cfg, "EMA_SLOPE_WINDOW_4H", 6)))
REALIZED_VOL_WINDOW_1H = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_1H", 24)))
REALIZED_VOL_WINDOW_4H = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_4H", 20)))
RANGE_WINDOW_4H = max(2, int(getattr(cfg, "RANGE_WINDOW_4H", 14)))
VWAP_WINDOW_4H = max(2, int(getattr(cfg, "VWAP_WINDOW_4H", 20)))
CHOPPINESS_WINDOW_1H = max(2, int(getattr(cfg, "CHOPPINESS_WINDOW_1H", 14)))
MARKET_CONTEXT_ZSCORE_WINDOW = max(5, int(getattr(cfg, "MARKET_CONTEXT_ZSCORE_WINDOW", 24)))
PRICE_ACTION_LEVEL_WINDOW_1H = max(4, int(getattr(cfg, "PRICE_ACTION_LEVEL_WINDOW_1H", 24)))
RANGE_COMPRESSION_SHORT_WINDOW_1H = max(2, int(getattr(cfg, "RANGE_COMPRESSION_SHORT_WINDOW_1H", 12)))
RANGE_COMPRESSION_LONG_WINDOW_1H = max(
    RANGE_COMPRESSION_SHORT_WINDOW_1H + 1,
    int(getattr(cfg, "RANGE_COMPRESSION_LONG_WINDOW_1H", 48)),
)
_thread_local = threading.local()

BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

# Интервалы в миллисекундах
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

BTC_REFERENCE_SYMBOL = "BTC/USDT"
BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FEATURE_OUTPUT_COLUMNS = [
    "return_1h_1",
    "return_1h_3",
    "return_1h_6",
    "return_1h_12",
    "return_1h_24",
    "realized_vol_1h",
    "ema_fast_slow",
    "linear_regression_slope_atr_1h_12",
    "linear_regression_slope_atr_1h_24",
    "price_zscore_1h",
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
    "volume_zscore_1h",
    # "parkinson_ratio",  # beta disabled
    # "choppiness_index",  # beta disabled
    # "volume_imbalance_1h_12",  # beta disabled
    "volume_ratio_1h_vs_4h",
    "price_position_1h",
    "distance_to_support_1h",
    "distance_to_resistance_1h",
    "price_position_4h",
    "adx_4h",
    "distance_to_rolling_high_4h",
    "distance_to_rolling_low_4h",
    "relative_strength_vs_btc_6h",
    "cross_sectional_rank_4h",
]
# Test Features V1: market-context / derivatives block
TEST_FEATURE_COLUMNS_V1 = [
    "premium_to_index_zscore",
    "mark_to_index_spread_zscore",
    "oi_change_1h",
    "oi_zscore_24",
    "funding_zscore_24",
    "oi_price_divergence",
]
if bool(getattr(cfg, "ENABLE_TEST_MARKET_CONTEXT_FEATURES", False)):
    FEATURE_OUTPUT_COLUMNS.extend(TEST_FEATURE_COLUMNS_V1)

# Test Features V2: price-action / level-behavior block
TEST_FEATURE_COLUMNS_V2 = [
    "rejection_strength_1h",
    "liquidity_sweep_proxy_1h",
    "range_compression_1h",
    "distance_to_session_high_1h",
    "distance_to_session_low_1h",
]
if bool(getattr(cfg, "ENABLE_TEST_PRICE_ACTION_FEATURES", False)):
    FEATURE_OUTPUT_COLUMNS.extend(TEST_FEATURE_COLUMNS_V2)
OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + FEATURE_OUTPUT_COLUMNS + ["Target", "MFE_long", "MFE_short"]


def init_db():
    """Создание таблицы если не существует"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            timeframe TEXT,
            open_time INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_volume REAL,
            PRIMARY KEY (symbol, timeframe, open_time)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_context (
            symbol TEXT,
            timeframe TEXT,
            open_time INTEGER,
            mark_close REAL,
            index_close REAL,
            premium_close REAL,
            open_interest REAL,
            funding_rate REAL,
            long_short_ratio REAL,
            PRIMARY KEY (symbol, timeframe, open_time)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_sync_state (
            dataset TEXT,
            symbol TEXT,
            timeframe TEXT,
            empty_since_ts INTEGER,
            last_checked_ts INTEGER,
            PRIMARY KEY (dataset, symbol, timeframe)
        )
    """)
    conn.commit()
    return conn


def get_http_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=max(8, BYBIT_MAX_WORKERS * 2),
            pool_maxsize=max(8, BYBIT_MAX_WORKERS * 2),
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "mlV2-etl-bybit/1.0"})
        _thread_local.session = session
    return session


def build_request_windows(start_ts, end_ts, timeframe_ms, page_limit=BYBIT_LIMIT):
    if start_ts > end_ts:
        return []

    max_span_ms = timeframe_ms * max(page_limit - 1, 1)
    windows = []
    window_start = start_ts

    while window_start <= end_ts:
        window_end = min(window_start + max_span_ms, end_ts)
        windows.append((window_start, window_end))
        window_start = window_end + timeframe_ms

    return windows


def request_bybit_json(url, params, request_name):
    last_error = None

    for attempt in range(BYBIT_RETRY_COUNT):
        try:
            response = get_http_session().get(url, params=params, timeout=BYBIT_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            ret_code = payload.get("retCode")
            if ret_code == 0:
                return payload

            last_error = RuntimeError(f"Bybit retCode={ret_code}, retMsg={payload.get('retMsg')}")
            if ret_code not in {10000, 10006, 10016}:
                raise last_error
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc

        if attempt + 1 < BYBIT_RETRY_COUNT:
            sleep_s = BYBIT_RETRY_SLEEP * (2 ** attempt)
            logger.warning(f"[{request_name}] retry {attempt + 1}/{BYBIT_RETRY_COUNT}: {last_error}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Bybit request failed for {request_name}: {last_error}")


def fetch_bybit_window(api_symbol, interval, window_start, window_end):
    params = {
        "category": BYBIT_CATEGORY,
        "symbol": api_symbol,
        "interval": interval,
        "start": window_start,
        "end": window_end,
        "limit": BYBIT_LIMIT,
    }
    payload = request_bybit_json(
        KLINE_URL,
        params,
        f"{api_symbol}-{interval}-{window_start}-{window_end}",
    )
    candles = payload.get("result", {}).get("list", [])
    rows = [
        (
            int(candle[0]),
            float(candle[1]),
            float(candle[2]),
            float(candle[3]),
            float(candle[4]),
            float(candle[5]),
            float(candle[6]),
        )
        for candle in candles
    ]
    rows.sort(key=lambda row: row[0])
    return rows


def flush_rows(conn, rows):
    if not rows:
        return 0

    rows.sort(key=lambda row: row[2])
    before_changes = conn.total_changes
    conn.executemany("INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn.total_changes - before_changes


def get_sync_state(conn, dataset, symbol, timeframe):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT empty_since_ts, last_checked_ts
        FROM data_sync_state
        WHERE dataset=? AND symbol=? AND timeframe=?
        """,
        (dataset, symbol, timeframe),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "empty_since_ts": row[0],
        "last_checked_ts": row[1],
    }


def upsert_sync_state(conn, dataset, symbol, timeframe, empty_since_ts, last_checked_ts):
    conn.execute(
        """
        INSERT INTO data_sync_state (dataset, symbol, timeframe, empty_since_ts, last_checked_ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(dataset, symbol, timeframe) DO UPDATE SET
            empty_since_ts = excluded.empty_since_ts,
            last_checked_ts = excluded.last_checked_ts
        """,
        (dataset, symbol, timeframe, empty_since_ts, last_checked_ts),
    )
    conn.commit()


def clear_sync_state(conn, dataset, symbol, timeframe):
    conn.execute(
        "DELETE FROM data_sync_state WHERE dataset=? AND symbol=? AND timeframe=?",
        (dataset, symbol, timeframe),
    )
    conn.commit()


def upsert_market_context_rows(conn, rows):
    if not rows:
        return 0

    rows.sort(key=lambda row: row[2])
    before_changes = conn.total_changes
    conn.executemany(
        """
        INSERT INTO market_context (
            symbol, timeframe, open_time, mark_close, index_close, premium_close,
            open_interest, funding_rate, long_short_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, open_time) DO UPDATE SET
            mark_close = COALESCE(excluded.mark_close, market_context.mark_close),
            index_close = COALESCE(excluded.index_close, market_context.index_close),
            premium_close = COALESCE(excluded.premium_close, market_context.premium_close),
            open_interest = COALESCE(excluded.open_interest, market_context.open_interest),
            funding_rate = COALESCE(excluded.funding_rate, market_context.funding_rate),
            long_short_ratio = COALESCE(excluded.long_short_ratio, market_context.long_short_ratio)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before_changes


def fetch_context_kline_series(url, api_symbol, timeframe, start_ts, end_ts):
    bybit_interval = BYBIT_INTERVALS.get(timeframe)
    timeframe_ms = TF_MS.get(timeframe)
    if bybit_interval is None or timeframe_ms is None:
        raise ValueError(f"Unsupported timeframe for market context: {timeframe}")

    windows = build_request_windows(start_ts, end_ts, timeframe_ms, page_limit=BYBIT_CONTEXT_LIMIT)
    rows = []
    for window_start, window_end in windows:
        payload = request_bybit_json(
            url,
            {
                "category": BYBIT_CATEGORY,
                "symbol": api_symbol,
                "interval": bybit_interval,
                "start": window_start,
                "end": window_end,
                "limit": BYBIT_CONTEXT_LIMIT,
            },
            f"context-kline-{api_symbol}-{timeframe}-{window_start}-{window_end}",
        )
        rows.extend(payload.get("result", {}).get("list", []))
    return rows


def fetch_open_interest_series(api_symbol, timeframe, start_ts, end_ts):
    timeframe_ms = TF_MS.get(timeframe)
    interval_time = timeframe
    windows = build_request_windows(start_ts, end_ts, timeframe_ms, page_limit=BYBIT_OI_LIMIT)
    rows = []
    for window_start, window_end in windows:
        payload = request_bybit_json(
            OPEN_INTEREST_URL,
            {
                "category": BYBIT_CATEGORY,
                "symbol": api_symbol,
                "intervalTime": interval_time,
                "startTime": window_start,
                "endTime": window_end,
                "limit": BYBIT_OI_LIMIT,
            },
            f"open-interest-{api_symbol}-{timeframe}-{window_start}-{window_end}",
        )
        rows.extend(payload.get("result", {}).get("list", []))
    return rows


def fetch_account_ratio_series(api_symbol, timeframe, start_ts, end_ts):
    timeframe_ms = TF_MS.get(timeframe)
    period = timeframe
    windows = build_request_windows(start_ts, end_ts, timeframe_ms, page_limit=BYBIT_RATIO_LIMIT)
    rows = []
    for window_start, window_end in windows:
        payload = request_bybit_json(
            ACCOUNT_RATIO_URL,
            {
                "category": BYBIT_CATEGORY,
                "symbol": api_symbol,
                "period": period,
                "startTime": window_start,
                "endTime": window_end,
                "limit": BYBIT_RATIO_LIMIT,
            },
            f"account-ratio-{api_symbol}-{timeframe}-{window_start}-{window_end}",
        )
        rows.extend(payload.get("result", {}).get("list", []))
    return rows


def fetch_funding_series(api_symbol, start_ts, end_ts):
    rows = []
    current_end = end_ts

    while True:
        payload = request_bybit_json(
            FUNDING_HISTORY_URL,
            {
                "category": BYBIT_CATEGORY,
                "symbol": api_symbol,
                "endTime": current_end,
                "limit": BYBIT_FUNDING_LIMIT,
            },
            f"funding-history-{api_symbol}-{current_end}",
        )
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break

        rows.extend(batch)
        min_ts = min(int(item["fundingRateTimestamp"]) for item in batch)
        if min_ts <= start_ts:
            break
        current_end = min_ts - 1

    return [item for item in rows if int(item["fundingRateTimestamp"]) >= start_ts]


def fetch_market_context(conn, symbol, timeframe):
    api_symbol = symbol.replace("/", "")
    timeframe_ms = TF_MS.get(timeframe)
    if timeframe_ms is None:
        raise ValueError(f"Unsupported timeframe for market context: {timeframe}")

    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM market_context WHERE symbol=? AND timeframe=?", (symbol, timeframe))
    last_ts = cur.fetchone()[0]

    if last_ts:
        start_ts = last_ts + timeframe_ms
    else:
        start_ts = int(datetime.fromisoformat(START_DATE).timestamp() * 1000)

    end_ts = int(datetime.fromisoformat(END_DATE).timestamp() * 1000) if END_DATE else int(time.time() * 1000)
    if start_ts > end_ts:
        logger.info(f"[{symbol}-{timeframe}] market context is already loaded up to {END_DATE}")
        return 0

    context_map = {}

    def ensure_row(ts):
        if ts not in context_map:
            context_map[ts] = {
                "mark_close": None,
                "index_close": None,
                "premium_close": None,
                "open_interest": None,
                "funding_rate": None,
                "long_short_ratio": None,
            }
        return context_map[ts]

    for candle in fetch_context_kline_series(MARK_KLINE_URL, api_symbol, timeframe, start_ts, end_ts):
        ensure_row(int(candle[0]))["mark_close"] = float(candle[4])

    for candle in fetch_context_kline_series(INDEX_KLINE_URL, api_symbol, timeframe, start_ts, end_ts):
        ensure_row(int(candle[0]))["index_close"] = float(candle[4])

    for candle in fetch_context_kline_series(PREMIUM_KLINE_URL, api_symbol, timeframe, start_ts, end_ts):
        ensure_row(int(candle[0]))["premium_close"] = float(candle[4])

    for item in fetch_open_interest_series(api_symbol, timeframe, start_ts, end_ts):
        ensure_row(int(item["timestamp"]))["open_interest"] = float(item["openInterest"])

    for item in fetch_funding_series(api_symbol, start_ts, end_ts):
        ensure_row(int(item["fundingRateTimestamp"]))["funding_rate"] = float(item["fundingRate"])

    for item in fetch_account_ratio_series(api_symbol, timeframe, start_ts, end_ts):
        ts = int(item["timestamp"])
        buy_ratio = float(item["buyRatio"])
        sell_ratio = float(item["sellRatio"])
        ensure_row(ts)["long_short_ratio"] = buy_ratio / sell_ratio if sell_ratio != 0 else np.nan

    rows = [
        (
            symbol,
            timeframe,
            ts,
            values["mark_close"],
            values["index_close"],
            values["premium_close"],
            values["open_interest"],
            values["funding_rate"],
            values["long_short_ratio"],
        )
        for ts, values in context_map.items()
    ]

    loaded = upsert_market_context_rows(conn, rows)
    logger.info(f"[{symbol}-{timeframe}] market context rows upserted: {loaded}")
    return loaded


def fetch_data(conn, symbol, timeframe):
    """
    Load candles from Bybit V5 starting from START_DATE or from the last saved candle.
    Supports incremental loading and parallel backfill over independent time windows.
    """
    api_symbol = symbol.replace("/", "")
    bybit_interval = BYBIT_INTERVALS.get(timeframe)
    if bybit_interval is None:
        raise ValueError(f"Unsupported timeframe for Bybit: {timeframe}")

    timeframe_ms = TF_MS.get(timeframe)
    if timeframe_ms is None:
        raise ValueError(f"timeframe {timeframe} is missing in TF_MS")

    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe))
    last_ts = cur.fetchone()[0]

    if last_ts:
        start_ts = last_ts + timeframe_ms
    else:
        start_ts = int(datetime.fromisoformat(START_DATE).timestamp() * 1000)

    end_ts = int(datetime.fromisoformat(END_DATE).timestamp() * 1000) if END_DATE else int(time.time() * 1000)
    sync_state = get_sync_state(conn, dataset="candles", symbol=symbol, timeframe=timeframe)
    if sync_state and sync_state.get("empty_since_ts") is not None:
        empty_since_ts = int(sync_state["empty_since_ts"])
        last_checked_ts = int(sync_state.get("last_checked_ts") or end_ts)
        next_retry_ts = last_checked_ts + timeframe_ms
        if start_ts >= empty_since_ts and end_ts < next_retry_ts:
            logger.info(
                f"[{symbol}-{timeframe}] no newer candles after "
                f"{datetime.fromtimestamp(empty_since_ts / 1000)}; "
                f"skipping repeated empty backfill until {datetime.fromtimestamp(next_retry_ts / 1000)}"
            )
            return 0

    if start_ts > end_ts:
        logger.info(f"[{symbol}-{timeframe}] data is already loaded up to {END_DATE}")
        return 0

    windows = build_request_windows(start_ts, end_ts, timeframe_ms)
    if not windows:
        return 0

    total_loaded = 0
    pending_rows = []
    fetched_any_candles = False
    total_windows = len(windows)
    progress_step = max(1, total_windows // 10)

    logger.info(
        f"[{symbol}-{timeframe}] Bybit backfill: {total_windows} windows, "
        f"limit={BYBIT_LIMIT}, workers={min(BYBIT_MAX_WORKERS, total_windows)}"
    )

    try:
        with ThreadPoolExecutor(max_workers=min(BYBIT_MAX_WORKERS, total_windows)) as executor:
            future_to_window = {
                executor.submit(fetch_bybit_window, api_symbol, bybit_interval, window_start, window_end): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                candles = future.result()

                if candles:
                    fetched_any_candles = True
                    pending_rows.extend(
                        (
                            symbol,
                            timeframe,
                            candle[0],
                            candle[1],
                            candle[2],
                            candle[3],
                            candle[4],
                            candle[5],
                            candle[6],
                        )
                        for candle in candles
                    )

                if len(pending_rows) >= DB_WRITE_BATCH_ROWS:
                    total_loaded += flush_rows(conn, pending_rows)
                    pending_rows.clear()

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = candles[-1][0] if candles else window_end
                    logger.info(
                        f"[{symbol}-{timeframe}] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )
    except Exception as exc:
        logger.error(f"load error for {symbol}-{timeframe}: {exc}")

    total_loaded += flush_rows(conn, pending_rows)
    if total_loaded > 0 or fetched_any_candles:
        clear_sync_state(conn, dataset="candles", symbol=symbol, timeframe=timeframe)
    else:
        upsert_sync_state(
            conn,
            dataset="candles",
            symbol=symbol,
            timeframe=timeframe,
            empty_since_ts=start_ts,
            last_checked_ts=end_ts,
        )
    return total_loaded


def load_from_db(conn, symbol, timeframe):
    """Загрузка данных из БД в DataFrame"""
    df = pd.read_sql_query(
        "SELECT open_time as timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY open_time",
        conn,
        params=(symbol, timeframe)
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def load_market_context_from_db(conn, symbol, timeframe):
    df = pd.read_sql_query(
        """
        SELECT
            open_time as timestamp,
            mark_close,
            index_close,
            premium_close,
            open_interest,
            funding_rate,
            long_short_ratio
        FROM market_context
        WHERE symbol=? AND timeframe=?
        ORDER BY open_time
        """,
        conn,
        params=(symbol, timeframe),
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


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


def add_test_price_action_features(df, atr_14):
    """Test Features V2: price-action / level-behavior block."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    prior_resistance_1h = high.shift(1).rolling(PRICE_ACTION_LEVEL_WINDOW_1H).max()
    prior_support_1h = low.shift(1).rolling(PRICE_ACTION_LEVEL_WINDOW_1H).min()

    bullish_rejection_1h = (prior_support_1h - low).clip(lower=0) * (close - prior_support_1h).clip(lower=0)
    bearish_rejection_1h = (high - prior_resistance_1h).clip(lower=0) * (prior_resistance_1h - close).clip(lower=0)
    df["rejection_strength_1h"] = safe_ratio(
        bullish_rejection_1h - bearish_rejection_1h,
        atr_14 * atr_14,
    )

    sweep_low_1h = safe_ratio((prior_support_1h - low).clip(lower=0), atr_14).where(
        (low < prior_support_1h) & (close > prior_support_1h),
        0.0,
    )
    sweep_high_1h = safe_ratio((high - prior_resistance_1h).clip(lower=0), atr_14).where(
        (high > prior_resistance_1h) & (close < prior_resistance_1h),
        0.0,
    )
    df["liquidity_sweep_proxy_1h"] = sweep_low_1h - sweep_high_1h

    range_short_1h = high.rolling(RANGE_COMPRESSION_SHORT_WINDOW_1H).max() - low.rolling(RANGE_COMPRESSION_SHORT_WINDOW_1H).min()
    range_long_1h = high.rolling(RANGE_COMPRESSION_LONG_WINDOW_1H).max() - low.rolling(RANGE_COMPRESSION_LONG_WINDOW_1H).min()
    df["range_compression_1h"] = safe_ratio(range_short_1h, range_long_1h)

    session_key = df["timestamp"].dt.floor("D")
    session_high_1h = high.groupby(session_key).cummax()
    session_low_1h = low.groupby(session_key).cummin()
    df["distance_to_session_high_1h"] = safe_ratio(session_high_1h - close, atr_14)
    df["distance_to_session_low_1h"] = safe_ratio(close - session_low_1h, atr_14)
    return df


def add_features(df):
    """Только 1H фичи из текущего набора."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    candle_range = (high - low).replace(0, np.nan)

    for period in (1, 3, 6, 12, 24):
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

    volume_mean_24 = volume.rolling(24).mean()
    volume_std_24 = volume.rolling(24).std().replace(0, np.nan)
    df["volume_zscore_1h"] = (volume - volume_mean_24) / volume_std_24
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
    if bool(getattr(cfg, "ENABLE_TEST_PRICE_ACTION_FEATURES", False)):
        df = add_test_price_action_features(df, atr_14)
    df["linear_regression_slope_atr_1h_12"] = safe_ratio(compute_linear_regression_slope(close, 12), atr_14)
    df["linear_regression_slope_atr_1h_24"] = safe_ratio(compute_linear_regression_slope(close, 24), atr_14)
    close_mean_24 = close.rolling(24).mean()
    close_std_24 = close.rolling(24).std().replace(0, np.nan)
    df["price_zscore_1h"] = (close - close_mean_24) / close_std_24
    df["volume_mean_3_1h"] = volume.rolling(3).mean()
    return df


def build_htf_feature_frame(htf_df, symbol):
    """Фичи 4H со сдвигом на одну свечу, чтобы не смотреть в незакрытый HTF-бар."""
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


def build_cross_sectional_rank(htf_feature_map):
    rank_frames = []
    for symbol, htf in htf_feature_map.items():
        frame = htf[["timestamp", "symbol", "return_4h_3"]].copy()
        frame = frame.dropna(subset=["return_4h_3"])
        if not frame.empty:
            rank_frames.append(frame)

    if not rank_frames:
        return {}

    rank_df = pd.concat(rank_frames, ignore_index=True)

    def normalize_rank(series):
        if len(series) == 1:
            return pd.Series(0.5, index=series.index)
        ranked = series.rank(method="average")
        return (ranked - 1) / (len(series) - 1)

    rank_df["cross_sectional_rank_4h"] = rank_df.groupby("timestamp")["return_4h_3"].transform(normalize_rank)

    rank_map = {}
    for symbol in htf_feature_map:
        symbol_rank_df = rank_df.loc[rank_df["symbol"] == symbol, ["timestamp", "cross_sectional_rank_4h"]].copy()
        rank_map[symbol] = symbol_rank_df
    return rank_map


def add_cross_sectional_rank(htf_feature_map):
    rank_map = build_cross_sectional_rank(htf_feature_map)
    enriched = {}
    for symbol, htf in htf_feature_map.items():
        rank_df = rank_map.get(symbol)
        merged = htf.copy()
        if rank_df is not None and not rank_df.empty:
            merged = merged.merge(rank_df, on="timestamp", how="left")
        else:
            merged["cross_sectional_rank_4h"] = np.nan
        enriched[symbol] = merged
    return enriched


def add_relative_strength_vs_btc(df, btc_df, symbol):
    df = df.copy()
    if btc_df is None or btc_df.empty:
        logger.warning("%s: BTC reference data is unavailable, relative strength will be NaN", symbol)
        df["relative_strength_vs_btc_6h"] = np.nan
        return df

    btc_reference = btc_df[["timestamp", "return_1h_6"]].rename(columns={"return_1h_6": "btc_return_1h_6"})
    df = df.merge(btc_reference, on="timestamp", how="left")
    df["relative_strength_vs_btc_6h"] = df["return_1h_6"] - df["btc_return_1h_6"]
    if symbol == BTC_REFERENCE_SYMBOL:
        df["relative_strength_vs_btc_6h"] = 0.0
    df.drop(columns=["btc_return_1h_6"], inplace=True)
    return df


def add_test_market_context_features(df, context_df, symbol):
    """Test Features V1: market-context / derivatives block."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    if context_df is None or context_df.empty:
        logger.warning("%s: market context is unavailable, test market-context features will be NaN", symbol)
        for column in TEST_FEATURE_COLUMNS_V1:
            df[column] = np.nan
        return df

    context = context_df.copy().sort_values("timestamp").reset_index(drop=True)
    df = pd.merge_asof(df, context, on="timestamp", direction="backward")
    context_columns = [
        "mark_close",
        "index_close",
        "premium_close",
        "open_interest",
        "funding_rate",
        "long_short_ratio",
    ]
    for column in context_columns:
        if column in df.columns:
            df[column] = df[column].ffill()

    spread_mark_index = safe_ratio(df["mark_close"], df["index_close"]) - 1.0
    spread_premium_index = safe_ratio(df["premium_close"], df["index_close"])

    spread_mark_mean = spread_mark_index.rolling(MARKET_CONTEXT_ZSCORE_WINDOW).mean()
    spread_mark_std = spread_mark_index.rolling(MARKET_CONTEXT_ZSCORE_WINDOW).std().replace(0, np.nan)
    df["mark_to_index_spread_zscore"] = (spread_mark_index - spread_mark_mean) / spread_mark_std

    spread_premium_mean = spread_premium_index.rolling(MARKET_CONTEXT_ZSCORE_WINDOW).mean()
    spread_premium_std = spread_premium_index.rolling(MARKET_CONTEXT_ZSCORE_WINDOW).std().replace(0, np.nan)
    df["premium_to_index_zscore"] = (spread_premium_index - spread_premium_mean) / spread_premium_std

    df["oi_change_1h"] = np.log(safe_ratio(df["open_interest"], df["open_interest"].shift(1)))
    oi_mean_24 = df["open_interest"].rolling(MARKET_CONTEXT_ZSCORE_WINDOW).mean()
    oi_std_24 = df["open_interest"].rolling(MARKET_CONTEXT_ZSCORE_WINDOW).std().replace(0, np.nan)
    df["oi_zscore_24"] = (df["open_interest"] - oi_mean_24) / oi_std_24

    funding_mean_24 = df["funding_rate"].rolling(MARKET_CONTEXT_ZSCORE_WINDOW).mean()
    funding_std_24 = df["funding_rate"].rolling(MARKET_CONTEXT_ZSCORE_WINDOW).std().replace(0, np.nan)
    df["funding_zscore_24"] = (df["funding_rate"] - funding_mean_24) / funding_std_24

    df["oi_price_divergence"] = df["oi_change_1h"] - df["return_1h_1"]

    df.drop(columns=[column for column in context_columns if column in df.columns], inplace=True)
    return df


def add_htf_features(df, htf_df):
    """
    Добавление только заданных 4H фичей.
    HTF-фрейм уже подготовлен со shift(1), чтобы не использовать незакрытую 4H свечу.
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
    df["volume_ratio_1h_vs_4h"] = safe_ratio(df["volume_mean_3_1h"], df["volume_mean_3_4h_per_hour"])
    df.drop(columns=["volume_mean_3_1h", "volume_mean_3_4h_per_hour"], inplace=True)
    return df


def compute_clean_pnl(direction, entry_price, exit_price):
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    return raw_pnl - (TAKER_COM + TAKER_COM)


def resolve_trade_exit(direction, entry_price, next_open, next_high, next_low):
    if direction == 1:
        stop_price = entry_price * (1 - SL_PCT)
        take_price = entry_price * (1 + TP_PCT)

        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - SLIPPAGE)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + SL_PCT)
        take_price = entry_price * (1 - TP_PCT)

        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + SLIPPAGE)
            return exit_price, "TP"

    return None, None


def simulate_trade_outcome(opens, highs, lows, start_idx, direction):
    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 + SLIPPAGE) if direction == 1 else base_open * (1 - SLIPPAGE)

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
        )
        if exit_price is not None:
            return compute_clean_pnl(direction, entry_price, exit_price), reason

    return 0.0, None


def triple_barrier_labeling(df):
    """Разметка данных (Teacher) — барьеры = фиксированные TP_PCT / SL_PCT.
    Если внутри одной свечи задеты оба барьера, приоритет всегда у SL."""
    labels = []

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    for i in range(len(df) - HORIZON):
        label = 0
        long_pnl, _ = simulate_trade_outcome(opens, highs, lows, i, direction=1)
        short_pnl, _ = simulate_trade_outcome(opens, highs, lows, i, direction=-1)

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1

        labels.append(label)

    labels.extend([0] * HORIZON)
    df['Target'] = labels
    return df


def add_mfe_targets(df):
    """MFE таргеты для регрессионной модели.
    MFE_long  = max % движение вверх за HORIZON свечей
    MFE_short = max % движение вниз за HORIZON свечей
    """
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    mfe_long = []
    mfe_short = []
    
    for i in range(len(df) - HORIZON):
        long_entry_price = opens[i + 1] * (1 + SLIPPAGE)
        short_entry_price = opens[i + 1] * (1 - SLIPPAGE)
        future_highs = highs[i + 1 : i + HORIZON + 1]
        future_lows = lows[i + 1 : i + HORIZON + 1]
        best_long_exit = future_highs.max() * (1 - SLIPPAGE)
        best_short_exit = future_lows.min() * (1 + SLIPPAGE)
        mfe_long.append(compute_clean_pnl(1, long_entry_price, best_long_exit))
        mfe_short.append(compute_clean_pnl(-1, short_entry_price, best_short_exit))
    
    # Последние HORIZON строк — фейковые (как в классификации)
    mfe_long.extend([0.0] * HORIZON)
    mfe_short.extend([0.0] * HORIZON)
    
    df['MFE_long'] = mfe_long
    df['MFE_short'] = mfe_short
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


def save_processed(df, symbol):
    """Сохранение обработанных данных в отдельную таблицу"""
    conn = sqlite3.connect(DB_PATH)
    table_name = symbol.replace('/', '_') + "_features"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"💾 {symbol} features сохранены ({len(df)} строк)")


def main():
    conn = init_db()
    enable_test_market_context_features = bool(getattr(cfg, "ENABLE_TEST_MARKET_CONTEXT_FEATURES", False))

    symbols_to_load = list(dict.fromkeys(SYMBOLS + [BTC_REFERENCE_SYMBOL]))
    for symbol in symbols_to_load:
        logger.info(f"Loading {symbol} {TIMEFRAME} from {START_DATE}...")
        loaded = fetch_data(conn, symbol, TIMEFRAME)
        logger.info(f"{symbol} {TIMEFRAME}: {loaded} new candles")

        logger.info(f"Loading {symbol} {HTF_TIMEFRAME} from {START_DATE}...")
        htf_loaded = fetch_data(conn, symbol, HTF_TIMEFRAME)
        logger.info(f"{symbol} {HTF_TIMEFRAME}: {htf_loaded} new candles")

        if enable_test_market_context_features:
            logger.info(f"Loading {symbol} market context {TIMEFRAME} from {START_DATE}...")
            context_loaded = fetch_market_context(conn, symbol, TIMEFRAME)
            logger.info(f"{symbol} market context {TIMEFRAME}: {context_loaded} new rows")

    base_1h_map = {}
    htf_feature_map = {}
    market_context_map = {}
    for symbol in symbols_to_load:
        df = load_from_db(conn, symbol, TIMEFRAME)
        htf_df = load_from_db(conn, symbol, HTF_TIMEFRAME)
        if df.empty or htf_df.empty:
            logger.warning(f"{symbol}: no data in DB (1h={len(df)}, 4h={len(htf_df)})")
            continue

        logger.info(f"{symbol}: 1h={len(df)}, 4h={len(htf_df)} rows")
        base_1h_map[symbol] = add_features(df)
        htf_feature_map[symbol] = build_htf_feature_frame(htf_df, symbol)
        if enable_test_market_context_features:
            market_context_map[symbol] = load_market_context_from_db(conn, symbol, TIMEFRAME)

    rank_source_map = {symbol: htf_feature_map[symbol] for symbol in SYMBOLS if symbol in htf_feature_map}
    ranked_map = add_cross_sectional_rank(rank_source_map)
    for symbol, ranked_htf in ranked_map.items():
        htf_feature_map[symbol] = ranked_htf
    for symbol in htf_feature_map:
        if symbol not in ranked_map:
            htf_feature_map[symbol]["cross_sectional_rank_4h"] = np.nan

    btc_df = base_1h_map.get(BTC_REFERENCE_SYMBOL)

    for symbol in SYMBOLS:
        df = base_1h_map.get(symbol)
        htf_df = htf_feature_map.get(symbol)
        if df is None or htf_df is None:
            logger.warning(f"{symbol}: skipped, missing prepared feature inputs")
            continue

        df = add_relative_strength_vs_btc(df, btc_df, symbol)
        df = add_htf_features(df, htf_df)
        if enable_test_market_context_features:
            df = add_test_market_context_features(df, market_context_map.get(symbol), symbol)
        df = triple_barrier_labeling(df)
        df = add_mfe_targets(df)
        df = finalize_feature_frame(df)
        save_processed(df, symbol)
        logger.info(f"{symbol}: saved {len(df)} rows with the requested feature set")

    conn.close()


if __name__ == '__main__':
    main()
