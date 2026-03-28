import sqlite3
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import etl
from config import *

# Futures settings come from config.py
TAKER_COM = globals().get("TAKER_COM", 0.0004)
MAKER_COM = globals().get("MAKER_COM", 0.0002)
SLIPPAGE = globals().get("SLIPPAGE", 0.0003)
LEVERAGE = globals().get("LEVERAGE", 1)
RISK_PER_TRADE = globals().get("RISK_PER_TRADE", 0.01)
DIRECTIONAL_PROBA_THRESHOLD = globals().get(
    "DIRECTIONAL_PROBA_THRESHOLD",
    globals().get("CONFIDENCE_THRESHOLD", 0.5),
)
BACKTEST_INITIAL_BALANCE = float(globals().get("BACKTEST_INITIAL_BALANCE", 100.0))
USE_DYNAMIC_BARRIERS = bool(globals().get("USE_DYNAMIC_BARRIERS", True))
BACKTEST_CHARTS_DIR = Path(globals().get("BACKTEST_CHARTS_DIR", "backtest_charts"))
BACKTEST_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_CURVE_PATH = BACKTEST_CHARTS_DIR / "equity_curve.png"

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


def get_end_date_cutoff():
    if not globals().get("END_DATE"):
        return None
    return pd.to_datetime(globals().get("END_DATE"), errors="coerce")


def apply_end_date_cutoff(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or timestamp_column not in df.columns:
        return df

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is None or pd.isna(end_cutoff):
        return df

    return df.loc[df[timestamp_column] <= end_cutoff].copy()

# LightGBM binary directional mapping (from train.py)
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TF_MS[timeframe]


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def format_pnl_pct(pnl_pct: float) -> str:
    color = ANSI_GREEN if pnl_pct >= 0 else ANSI_RED
    return colorize(f"{pnl_pct:+.2f}%", color)


def format_reason(reason: str) -> str:
    if reason == "TP":
        return colorize("✅ TP", ANSI_GREEN)
    if reason == "SL":
        return colorize("❌ SL", ANSI_RED)
    return reason


def build_entry_score(direction_prob: float, signal_gap: float) -> float:
    edge = max(0.0, direction_prob - DIRECTIONAL_PROBA_THRESHOLD)
    return edge * 10 + signal_gap


def format_reason(reason: str) -> str:
    if reason == "TP":
        return colorize("TP", ANSI_GREEN)
    if reason == "SL":
        return colorize("SL", ANSI_RED)
    return reason


def resolve_directional_signal(p_long: float, p_short: float) -> tuple[int, float, float]:
    signal_gap = abs(p_long - p_short)

    if (
        p_long >= DIRECTIONAL_PROBA_THRESHOLD
        and (p_long - p_short) >= MIN_SIGNAL_GAP
    ):
        return 1, p_long, signal_gap
    if (
        p_short >= DIRECTIONAL_PROBA_THRESHOLD
        and (p_short - p_long) >= MIN_SIGNAL_GAP
    ):
        return -1, p_short, signal_gap
    return 0, max(p_long, p_short), signal_gap


def get_barrier_pcts(feature_row: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None

    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None

    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct


def compact_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
    right_align = right_align or set()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(row_values):
        formatted = []
        for idx, cell in enumerate(row_values):
            text = str(cell)
            if idx in right_align:
                formatted.append(text.rjust(widths[idx]))
            else:
                formatted.append(text.ljust(widths[idx]))
        return "│ " + " │ ".join(formatted) + " │"

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    print(top)
    print(format_row(headers))
    print(mid)
    for row in rows:
        print(format_row(row))
    print(bottom)


def load_raw_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT open_time as timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol=? AND timeframe=?
        ORDER BY open_time
        """,
        conn,
        params=(symbol, timeframe),
    )
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    tf_ms = timeframe_to_ms(timeframe)
    df["close_time"] = df["timestamp"] + pd.to_timedelta(tf_ms, unit="ms")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = apply_end_date_cutoff(df)
    return df


def load_all_raw_data(symbols):
    all_data = {}

    print(f"Loading raw data for {len(symbols)} symbols...")
    for sym in symbols:
        try:
            df_main = load_raw_candles(sym, TIMEFRAME)
            df_htf = load_raw_candles(sym, HTF_TIMEFRAME)

            if df_main.empty:
                print(f"Warning: {sym} has no main TF data ({TIMEFRAME})")
                continue

            if df_htf.empty:
                print(f"Warning: {sym} has no HTF data ({HTF_TIMEFRAME})")
                continue

            all_data[sym] = {"main": df_main, "htf": df_htf}
            print(f"{sym}: main={len(df_main)} candles, htf={len(df_htf)} candles")
        except Exception as e:
            print(f"Warning: failed loading {sym}: {e}")

    return all_data


def load_precomputed_features(symbol: str, symbol_categories=None, required_columns: list | None = None) -> pd.DataFrame:
    table_name = symbol.replace("/", "_") + "_features"
    conn = sqlite3.connect(DB_PATH)
    try:
        select_columns = ["timestamp"]
        if required_columns:
            select_columns.extend(column for column in required_columns if column != "timestamp")
        else:
            select_columns.append("*")

        if select_columns[-1] == "*":
            query = f"SELECT * FROM {table_name}"
        else:
            quoted_columns = ", ".join(f'"{column}"' for column in dict.fromkeys(select_columns))
            query = f"SELECT {quoted_columns} FROM {table_name}"
        df = pd.read_sql_query(query, conn)
    except Exception:
        conn.close()
        return pd.DataFrame()
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = apply_end_date_cutoff(df)
    df["symbol"] = symbol
    if symbol_categories is None:
        df["symbol"] = df["symbol"].astype("category")
    else:
        df["symbol"] = pd.Categorical(df["symbol"], categories=symbol_categories)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_timestamp_index(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    return df.set_index("timestamp", drop=False).to_dict("index")


def get_common_main_timestamps(all_data: dict) -> list:
    if not all_data:
        return []

    ts_sets = [set(payload["main"]["timestamp"]) for payload in all_data.values()]
    return sorted(list(set.intersection(*ts_sets))) if ts_sets else []


def filter_symbols_with_recent_data(all_data: dict, min_common_candles: int = 300):
    if not all_data:
        return {}, []

    timeframe_delta = pd.to_timedelta(timeframe_to_ms(TIMEFRAME), unit="ms")
    latest_timestamp = max(payload["main"]["timestamp"].iloc[-1] for payload in all_data.values())
    recent_cutoff = latest_timestamp - (timeframe_delta * min_common_candles)

    filtered = {}
    dropped = []
    for symbol, payload in all_data.items():
        symbol_last_ts = payload["main"]["timestamp"].iloc[-1]
        if symbol_last_ts < recent_cutoff:
            dropped.append(symbol)
            continue
        filtered[symbol] = payload

    return filtered, dropped


def prepare_dataset_for_time(df: pd.DataFrame, analysis_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df[df["close_time"] <= analysis_ts].copy()
    if out.empty:
        return out

    return out.drop(columns=["close_time"], errors="ignore").reset_index(drop=True)


def build_feature_row_at_time(
    symbol: str,
    main_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    analysis_ts: pd.Timestamp,
    feature_names: list,
    symbol_categories=None,
):
    main_cut = prepare_dataset_for_time(main_df, analysis_ts)
    htf_cut = prepare_dataset_for_time(htf_df, analysis_ts)

    if main_cut is None or main_cut.empty or len(main_cut) < 250:
        return None

    if htf_cut is None or htf_cut.empty or len(htf_cut) < 60:
        return None

    try:
        feat_main = etl.add_features(main_cut)
        if feat_main is None or feat_main.empty:
            return None

        feat_main = etl.add_htf_features(feat_main, htf_cut)
        if feat_main is None or feat_main.empty:
            return None

        feat_main["symbol"] = symbol
        if symbol_categories is None:
            feat_main["symbol"] = feat_main["symbol"].astype("category")
        else:
            feat_main["symbol"] = pd.Categorical(feat_main["symbol"], categories=symbol_categories)
        latest_row = feat_main.iloc[[-1]].copy()
        missing = [f for f in feature_names if f not in latest_row.columns]
        if missing:
            print(f"Warning: {symbol} missing features: {missing[:10]}")
            return None

        if latest_row[feature_names].isna().any(axis=None):
            return None

        return latest_row
    except Exception as e:
        print(f"Warning: feature build failed for {symbol} @ {analysis_ts}: {e}")
        return None


def get_exec_row_by_ts(df: pd.DataFrame, ts: pd.Timestamp):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None
    return row.iloc[0]


def get_feature_row_precomputed(df: pd.DataFrame, ts: pd.Timestamp, feature_names: list):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None

    latest_row = row.iloc[[-1]].copy()
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None

    if latest_row[feature_names].isna().any(axis=None):
        return None

    return latest_row


def get_exec_row_by_ts_index(indexed_rows: dict, ts: pd.Timestamp):
    row = indexed_rows.get(ts)
    if row is None:
        return None
    return row


def get_feature_row_precomputed_index(indexed_rows: dict, ts: pd.Timestamp, feature_names: list):
    row = indexed_rows.get(ts)
    if row is None:
        return None

    latest_row = pd.DataFrame([row])
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None

    if latest_row[feature_names].isna().any(axis=None):
        return None

    return latest_row


def normalize_features_for_model(latest_row: pd.DataFrame, feature_names: list, symbol_categories=None):
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict):
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def prepare_precomputed_feature_store(
    feat_df: pd.DataFrame,
    feature_names: list,
    symbol_categories=None,
    clip_bounds: dict | None = None,
) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return pd.DataFrame(columns=feature_names)

    missing = [feature for feature in feature_names if feature not in feat_df.columns]
    if missing:
        return pd.DataFrame(columns=feature_names)

    timestamps = feat_df["timestamp"].copy()
    prepared = normalize_features_for_model(
        feat_df,
        feature_names,
        symbol_categories=symbol_categories,
    )
    prepared = apply_feature_clip_bounds(prepared, clip_bounds or {})
    prepared.insert(0, "timestamp", timestamps.values)
    prepared = prepared.dropna(subset=feature_names)
    prepared = prepared.drop_duplicates(subset=["timestamp"], keep="last")
    return prepared.set_index("timestamp", drop=True).sort_index()


def get_feature_batch_precomputed(
    feature_store: dict,
    symbols: list,
    ts: pd.Timestamp,
):
    batch_frames = []
    batch_symbols = []

    for symbol in symbols:
        prepared = feature_store.get(symbol)
        if prepared is None or prepared.empty or ts not in prepared.index:
            continue

        batch_frames.append(prepared.loc[[ts]])
        batch_symbols.append(symbol)

    if not batch_frames:
        return [], pd.DataFrame()

    return batch_symbols, pd.concat(batch_frames, axis=0)


def backtest():
    print("Loading model and features...")

    if not ALLOW_LONGS and not ALLOW_SHORTS:
        print("Error: both ALLOW_LONGS and ALLOW_SHORTS are disabled.")
        return

    # --- Load LightGBM model ---
    model_path = MODELS_DIR / "lightgbm_target.joblib"
    features_meta_path = MODELS_DIR / "lightgbm_target_features.json"

    if not model_path.exists():
        print(f"Error: model not found at {model_path}. Run train.py first.")
        return

    if not features_meta_path.exists():
        print(f"Error: features metadata not found at {features_meta_path}. Run train.py first.")
        return

    model = joblib.load(model_path)

    with open(features_meta_path, "r", encoding="utf-8") as f:
        features_meta = json.load(f)

    feature_names = features_meta["feature_columns"]
    trained_symbols = list(features_meta.get("symbols", SYMBOLS))
    use_symbol_feature = "symbol" in feature_names
    unseen_symbols = [symbol for symbol in SYMBOLS if symbol not in trained_symbols]
    if use_symbol_feature:
        symbol_categories = list(dict.fromkeys(trained_symbols + list(SYMBOLS)))
        if unseen_symbols:
            print(
                "Warning: backtest includes symbols absent from training metadata: "
                + ", ".join(unseen_symbols)
            )
    else:
        symbol_categories = None
    feature_clip_meta = features_meta.get("feature_clip", {})
    clip_bounds = feature_clip_meta.get("bounds", {})
    print(f"Loaded LightGBM model with {len(feature_names)} features")
    if clip_bounds:
        print(
            "Feature clipping: "
            f"{len(clip_bounds)} columns "
            f"[{feature_clip_meta.get('lower_q', 0.01) * 100:.2f}%, "
            f"{feature_clip_meta.get('upper_q', 0.99) * 100:.2f}%]"
        )

    all_raw = load_all_raw_data(SYMBOLS)
    if not all_raw:
        print("Error: no raw data for backtest.")
        return

    print(
        "Feature mode: "
        + ("realtime rebuild" if BACKTEST_REALTIME_FEATURES else "precomputed DB features")
    )

    all_features = {}
    all_main_index = {}
    if not BACKTEST_REALTIME_FEATURES:
        for sym in list(all_raw.keys()):
            feat_df = load_precomputed_features(
                sym,
                symbol_categories=symbol_categories,
                required_columns=feature_names + ["barrier_stop_pct", "barrier_take_pct"],
            )
            if feat_df.empty:
                print(f"Warning: {sym} has no precomputed features table.")
                continue
            all_features[sym] = feat_df

        all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features}
        if not all_raw:
            print("Error: no symbols with precomputed features available.")
            return

    for sym, payload in all_raw.items():
        all_main_index[sym] = build_timestamp_index(payload["main"])

    all_features_prepared = {}
    if not BACKTEST_REALTIME_FEATURES:
        for sym, feat_df in all_features.items():
            prepared = prepare_precomputed_feature_store(
                feat_df,
                feature_names,
                symbol_categories=symbol_categories,
                clip_bounds=clip_bounds,
            )
            if prepared.empty:
                print(f"Warning: {sym} has no usable precomputed feature rows after preparation.")
                continue
            all_features_prepared[sym] = prepared

        all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features_prepared}
        if not all_raw:
            print("Error: no symbols with prepared precomputed features available.")
            return

    all_raw, stale_symbols = filter_symbols_with_recent_data(all_raw, min_common_candles=300)
    if stale_symbols:
        print(
            "Warning: dropped symbols without enough recent overlap for the backtest window: "
            + ", ".join(stale_symbols)
        )
    if not all_raw:
        print("Error: no symbols with enough recent overlap for backtest.")
        return

    common_timestamps = get_common_main_timestamps(all_raw)
    if len(common_timestamps) < 300:
        print("Error: too few common timestamps for test.")
        return

    split_idx = int(len(common_timestamps) * 0.85)
    test_timestamps = common_timestamps[split_idx:]

    if len(test_timestamps) < 2:
        print("Error: too little data for test period.")
        return

    balance = BACKTEST_INITIAL_BALANCE
    initial_balance = balance
    positions = {sym: None for sym in all_raw}
    trades = []
    equity_curve = []
    equity_timestamps = []
    monthly_stats = {}
    peak_balance = balance
    max_drawdown = 0.0
    used_margin = 0.0

    print("\n📋 Backtest Configuration:")
    print(f"   Period: {test_timestamps[0].isoformat()} to {test_timestamps[-1].isoformat()}")
    print(f"   Symbols: {', '.join(compact_symbol(sym) for sym in all_raw.keys())}")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Risk per Trade: {RISK_PER_TRADE * 100:.0f}%")
    print(f"   Leverage: {LEVERAGE:.0f}x")
    print(f"   Main TF: {TIMEFRAME} | HTF: {HTF_TIMEFRAME}")
    if ALLOW_LONGS and ALLOW_SHORTS:
        direction_mode = "LONG+SHORT"
    elif ALLOW_LONGS:
        direction_mode = "LONG ONLY"
    else:
        direction_mode = "SHORT ONLY"
    print(f"   Direction mode: {direction_mode}")
    print(f"   Directional probability threshold: {DIRECTIONAL_PROBA_THRESHOLD:.2f}")
    print(f"   Min signal gap: {MIN_SIGNAL_GAP:.2f}")
    print(f"   Batch entries per bar: {BACKTEST_MAX_NEW_POSITIONS_PER_BAR}")
    print(f"   Max open positions: {BACKTEST_MAX_OPEN_POSITIONS}")
    if USE_DYNAMIC_BARRIERS:
        print(
            "   Dynamic barriers: "
            f"ATRx{globals().get('BARRIER_ATR_MULTIPLIER', 1.25):.2f}, "
            f"RVOLx{globals().get('BARRIER_RVOL_MULTIPLIER', 0.75):.2f}, "
            f"TP/SL={globals().get('BARRIER_TP_TO_SL_RATIO', 2.0):.2f}"
        )
    else:
        print(f"   TP: {TP_PCT:.4f} | SL: {SL_PCT:.4f}")
    print(f"\nBacktest on {len(test_timestamps)} candles")
    print("-" * 80)

    num_candles = len(test_timestamps)

    for i in range(num_candles - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i + 1]

        equity_curve.append(balance)
        equity_timestamps.append(current_ts)

        month_key = next_ts.strftime("%Y-%m")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {
                "pnl_abs": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "start_balance": balance,
            }

        market_batch = {}
        for sym, payload in all_raw.items():
            curr_exec = get_exec_row_by_ts_index(all_main_index[sym], current_ts)
            next_exec = get_exec_row_by_ts_index(all_main_index[sym], next_ts)
            if curr_exec is None or next_exec is None:
                continue

            market_batch[sym] = {
                "main": payload["main"],
                "htf": payload["htf"],
                "next_open": float(next_exec["open"]),
                "next_high": float(next_exec["high"]),
                "next_low": float(next_exec["low"]),
            }

        # Phase 1: all exits are evaluated on the same market snapshot.
        for sym, ctx in market_batch.items():
            if positions[sym] is None:
                continue

            pos = positions[sym]
            entry_price = pos["entry"]
            direction = pos["dir"]
            position_notional = pos["size"]
            stop_pct = pos["stop_pct"]
            take_pct = pos["take_pct"]
            next_open = ctx["next_open"]
            next_high = ctx["next_high"]
            next_low = ctx["next_low"]
            exit_signal = False
            exit_price = 0.0
            reason = ""

            if direction == 1:
                stop_price = entry_price * (1 - stop_pct)
                take_price = entry_price * (1 + take_pct)

                if next_low <= stop_price:
                    exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
                    exit_signal = True
                    reason = "SL"
                elif next_high >= take_price:
                    exit_price = take_price * (1 - SLIPPAGE)
                    exit_signal = True
                    reason = "TP"
            else:
                stop_price = entry_price * (1 + stop_pct)
                take_price = entry_price * (1 - take_pct)

                if next_high >= stop_price:
                    exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                    exit_signal = True
                    reason = "SL"
                elif next_low <= take_price:
                    exit_price = take_price * (1 + SLIPPAGE)
                    exit_signal = True
                    reason = "TP"

            if not exit_signal:
                continue

            if direction == 1:
                raw_pnl = (exit_price - entry_price) / entry_price
            else:
                raw_pnl = (entry_price - exit_price) / entry_price

            commission = position_notional * (TAKER_COM + TAKER_COM)
            pnl_clean = raw_pnl - (TAKER_COM + TAKER_COM)
            trade_profit = position_notional * pnl_clean

            used_margin -= pos["margin"]
            if used_margin < 0:
                used_margin = 0.0

            balance += trade_profit

            trades.append(
                {
                    "sym": sym,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "reason": reason,
                    "pnl_pct": pnl_clean,
                    "pnl_abs": trade_profit,
                    "commission": commission,
                    "ts": next_ts,
                }
            )
            monthly_stats[month_key]["pnl_abs"] += trade_profit
            monthly_stats[month_key]["trades"] += 1
            if pnl_clean > 0:
                monthly_stats[month_key]["wins"] += 1
            else:
                monthly_stats[month_key]["losses"] += 1

            if balance > peak_balance:
                peak_balance = balance

            current_dd = (peak_balance - balance) / peak_balance * 100
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            positions[sym] = None

            exit_icon = "\u274C" if reason == "SL" else "\u2705" if reason == "TP" else "\u2139\uFE0F"
            print(
                f"[{next_ts}] {exit_icon} {sym}: {format_reason(reason)} | "
                f"PnL: {format_pnl_pct(pnl_clean * 100)} | "
                f"Com: {commission:.2f}$ | "
                f"Bal: {balance:.2f}"
            )

        # Phase 2: collect all entry candidates first, then rank them.
        snapshot_balance = balance
        entry_candidates = []
        if BACKTEST_REALTIME_FEATURES:
            for sym, ctx in market_batch.items():
                if positions[sym] is not None:
                    continue

                latest_row = build_feature_row_at_time(
                    symbol=sym,
                    main_df=ctx["main"],
                    htf_df=ctx["htf"],
                    analysis_ts=next_ts,
                    feature_names=feature_names,
                    symbol_categories=symbol_categories,
                )

                if latest_row is None or latest_row.empty:
                    continue

                current_features = normalize_features_for_model(
                    latest_row,
                    feature_names,
                    symbol_categories=symbol_categories,
                )
                current_features = apply_feature_clip_bounds(current_features, clip_bounds)
                stop_pct, take_pct = get_barrier_pcts(latest_row)
                if stop_pct is None or take_pct is None:
                    continue

                proba = model.predict_proba(current_features)[0]
                p_short = float(proba[0])
                p_long = float(proba[1])

                signal, direction_prob, signal_gap = resolve_directional_signal(p_long, p_short)

                if signal == 0:
                    continue

                if signal == 1 and not ALLOW_LONGS:
                    continue
                if signal == -1 and not ALLOW_SHORTS:
                    continue

                if signal == 1:
                    direction_str = "LONG"
                    entry_price = ctx["next_open"] * (1 + SLIPPAGE)
                else:
                    direction_str = "SHORT"
                    entry_price = ctx["next_open"] * (1 - SLIPPAGE)

                risk_capital = snapshot_balance * RISK_PER_TRADE
                position_notional = min(risk_capital / stop_pct, snapshot_balance * LEVERAGE)
                required_margin = position_notional / LEVERAGE

                if position_notional < 10:
                    continue

                entry_candidates.append(
                    {
                        "sym": sym,
                        "signal": signal,
                        "direction_str": direction_str,
                        "entry_price": entry_price,
                        "position_notional": position_notional,
                        "required_margin": required_margin,
                        "stop_pct": stop_pct,
                        "take_pct": take_pct,
                        "p_long": p_long,
                        "p_short": p_short,
                        "direction_prob": direction_prob,
                        "score": build_entry_score(direction_prob, signal_gap),
                    }
                )
        else:
            candidate_symbols = [
                sym for sym in market_batch
                if positions[sym] is None and sym in all_features_prepared
            ]
            batch_symbols, batch_features = get_feature_batch_precomputed(
                all_features_prepared,
                candidate_symbols,
                current_ts,
            )
            if not batch_features.empty:
                batch_proba = model.predict_proba(batch_features[feature_names])
                for sym, proba in zip(batch_symbols, batch_proba):
                    ctx = market_batch[sym]
                    feature_row = get_feature_row_precomputed(
                        all_features[sym],
                        current_ts,
                        feature_names + ["barrier_stop_pct", "barrier_take_pct"],
                    )
                    stop_pct, take_pct = get_barrier_pcts(feature_row)
                    if stop_pct is None or take_pct is None:
                        continue
                    p_short = float(proba[0])
                    p_long = float(proba[1])

                    signal, direction_prob, signal_gap = resolve_directional_signal(p_long, p_short)

                    if signal == 0:
                        continue

                    if signal == 1 and not ALLOW_LONGS:
                        continue
                    if signal == -1 and not ALLOW_SHORTS:
                        continue

                    if signal == 1:
                        direction_str = "LONG"
                        entry_price = ctx["next_open"] * (1 + SLIPPAGE)
                    else:
                        direction_str = "SHORT"
                        entry_price = ctx["next_open"] * (1 - SLIPPAGE)

                    risk_capital = snapshot_balance * RISK_PER_TRADE
                    position_notional = min(risk_capital / stop_pct, snapshot_balance * LEVERAGE)
                    required_margin = position_notional / LEVERAGE

                    if position_notional < 10:
                        continue

                    entry_candidates.append(
                        {
                            "sym": sym,
                            "signal": signal,
                            "direction_str": direction_str,
                            "entry_price": entry_price,
                            "position_notional": position_notional,
                            "required_margin": required_margin,
                            "stop_pct": stop_pct,
                            "take_pct": take_pct,
                            "p_long": p_long,
                            "p_short": p_short,
                            "direction_prob": direction_prob,
                            "score": build_entry_score(direction_prob, signal_gap),
                        }
                    )

        if not entry_candidates:
            continue

        entry_candidates.sort(
            key=lambda candidate: (
                candidate["score"],
                candidate["direction_prob"],
            ),
            reverse=True,
        )

        opened_this_bar = 0
        open_positions_count = sum(pos is not None for pos in positions.values())
        for candidate in entry_candidates:
            if opened_this_bar >= BACKTEST_MAX_NEW_POSITIONS_PER_BAR:
                break
            if open_positions_count >= BACKTEST_MAX_OPEN_POSITIONS:
                break

            available_balance = balance - used_margin
            if available_balance <= 0:
                break

            required_margin = min(candidate["required_margin"], available_balance)
            position_notional = min(candidate["position_notional"], required_margin * LEVERAGE)

            if position_notional < 10 or required_margin <= 0:
                continue

            used_margin += required_margin
            positions[candidate["sym"]] = {
                "dir": candidate["signal"],
                "entry": candidate["entry_price"],
                "size": position_notional,
                "margin": required_margin,
                "stop_pct": candidate["stop_pct"],
                "take_pct": candidate["take_pct"],
                "ts_open": next_ts,
            }
            opened_this_bar += 1
            open_positions_count += 1

            print(
                f"[{next_ts}] \U0001F525 OPEN {candidate['direction_str']}: {candidate['sym']} "
                f"(Long={candidate['p_long']:.2f}, Short={candidate['p_short']:.2f}, "
                f"Score={candidate['score']:.3f}) "
                f"at {candidate['entry_price']:.4f} | "
                f"Size: {position_notional:.2f}$ "
                f"Margin: {required_margin:.2f}$"
            )

    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve, index=equity_timestamps)
        daily_equity = equity_series.resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
            cagr = ((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1) if total_days > 0 else 0
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe = (mean_daily_return / std_daily_return) * np.sqrt(365)

            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = (mean_daily_return / downside_returns.std()) * np.sqrt(365)
            else:
                sortino = 0.0

            calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0
            calmar = 0.0
            cagr = 0.0

        if trades:
            returns = np.array([t["pnl_abs"] for t in trades])
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            pf = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0
        cagr = 0.0
        pf = 0.0

    total_trades = len(trades)
    total_wins = sum(1 for trade in trades if trade["pnl_abs"] > 0)
    total_losses = total_trades - total_wins
    total_pnl_abs = sum(trade["pnl_abs"] for trade in trades)
    total_fees = sum(trade.get("commission", 0.0) for trade in trades)
    final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
    expectancy = (total_pnl_abs / total_trades) if total_trades > 0 else 0.0

    print("\nSimulation finished.\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║           PORTFOLIO BACKTEST RESULTS                     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\n📊 Trades: {total_trades} (W: {total_wins} / L: {total_losses})")
    print("💰 Equity:")
    print(f"   Start: ${initial_balance:.2f}")
    print(f"   End:   ${balance:.2f}")
    print(f"   PnL:   {format_signed_dollars(total_pnl_abs)} ({format_percent_value(total_return_pct)})")
    print(f"   Fees:  ${total_fees:.2f}")
    print("📉 Risk:")
    print(f"   Max DD: {max_drawdown:.2f}%")
    print(f"   Profit Factor: {pf:.2f}")
    print(f"   Expectancy: {format_signed_dollars(expectancy)}")
    print(f"   Sharpe: {sharpe:.2f}")

    monthly_rows = []
    for month_key in sorted(monthly_stats.keys()):
        stats = monthly_stats[month_key]
        monthly_rows.append(
            [
                month_key,
                format_signed_dollars(stats["pnl_abs"]),
                str(stats["trades"]),
                str(stats["wins"]),
                str(stats["losses"]),
            ]
        )

    if monthly_rows:
        print("\n📅 Monthly Performance Extended:")
        print_table(
            ["Month", "PnL", "Total", "Wins", "Losses"],
            monthly_rows,
            right_align={1, 2, 3, 4},
        )

    symbol_stats = {}
    for trade in trades:
        stats = symbol_stats.setdefault(
            trade["sym"],
            {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        )
        stats["trades"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["reason"] == "TP":
            stats["tp"] += 1
        if trade["reason"] == "SL":
            stats["sl"] += 1
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    if symbol_stats:
        coin_rows = []
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            winrate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0.0
            coin_rows.append(
                [
                    compact_symbol(symbol),
                    str(stats["trades"]),
                    str(stats["tp"]),
                    str(stats["sl"]),
                    f"{winrate:.1f}%",
                    format_signed_dollars(stats["pnl_abs"]),
                ]
            )

        print("\n📊 Summary by Coin:")
        print_table(
            ["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"],
            coin_rows,
            right_align={1, 2, 3, 4, 5},
        )

    direction_stats = {
        "LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        "SHORT": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
    }
    for trade in trades:
        stats = direction_stats[trade["direction"]]
        stats["total"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    direction_rows = []
    for direction in ("LONG", "SHORT"):
        stats = direction_stats[direction]
        direction_rows.append(
            [
                direction,
                str(stats["total"]),
                str(stats["wins"]),
                str(stats["losses"]),
                format_signed_dollars(stats["pnl_abs"]),
            ]
        )

    print("\n📈 Long / Short Summary:")
    print_table(
        ["Direction", "Total", "Wins", "Losses", "PnL"],
        direction_rows,
        right_align={1, 2, 3, 4},
    )

    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, label="Portfolio Equity")
        plt.axhline(y=initial_balance, linestyle="--")
        plt.title(f"Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(EQUITY_CURVE_PATH, dpi=150)
        plt.show()
        print(f"\nSaved chart: {EQUITY_CURVE_PATH}")


if __name__ == "__main__":
    backtest()
