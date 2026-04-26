"""Оркестратор и цикл бэктеста портфеля (логика перенесена из bt.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import *
from src.application.backtest import backtest_data as ld
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

# Реэкспорт для bt.py и скриптов
TF_MS = ld.TF_MS
DEFAULT_MODEL_NAME = ld.DEFAULT_MODEL_NAME
get_end_date_cutoff = ld.get_end_date_cutoff
apply_end_date_cutoff = ld.apply_end_date_cutoff
parse_period_payload = ld.parse_period_payload
timeframe_to_ms = ld.timeframe_to_ms
load_raw_candles = ld.load_raw_candles
load_all_raw_data = ld.load_all_raw_data
load_precomputed_features = ld.load_precomputed_features
build_timestamp_index = ld.build_timestamp_index
get_common_main_timestamps = ld.get_common_main_timestamps
filter_symbols_with_recent_data = ld.filter_symbols_with_recent_data
filter_symbols_with_period_overlap = ld.filter_symbols_with_period_overlap
prepare_dataset_for_time = ld.prepare_dataset_for_time
build_feature_row_at_time = ld.build_feature_row_at_time
get_exec_row_by_ts = ld.get_exec_row_by_ts
get_feature_row_precomputed = ld.get_feature_row_precomputed
get_exec_row_by_ts_index = ld.get_exec_row_by_ts_index
get_feature_row_precomputed_index = ld.get_feature_row_precomputed_index
is_candidate_event = ld.is_candidate_event
normalize_features_for_model = ld.normalize_features_for_model
apply_feature_clip_bounds = ld.apply_feature_clip_bounds
prepare_precomputed_feature_store = ld.prepare_precomputed_feature_store
get_feature_batch_precomputed = ld.get_feature_batch_precomputed
build_prediction_lookup = ld.build_prediction_lookup
get_barrier_pcts = ld.get_barrier_pcts

# Futures settings come from config.py
TAKER_COM = globals().get("TAKER_COM", 0.0004)
MAKER_COM = globals().get("MAKER_COM", 0.0002)
SLIPPAGE = globals().get("SLIPPAGE", 0.0003)
LEVERAGE = globals().get("LEVERAGE", 1)
RISK_PER_TRADE = globals().get("RISK_PER_TRADE", 0.01)
BACKTEST_SL_COOLDOWN_BARS = int(globals().get("BACKTEST_SL_COOLDOWN_BARS", 0))
BACKTEST_MAX_SL_PER_DAY = int(globals().get("BACKTEST_MAX_SL_PER_DAY", 0))
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = int(
    globals().get("BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0)
)
BACKTEST_REDUCED_RISK_PER_TRADE = float(globals().get("BACKTEST_REDUCED_RISK_PER_TRADE", RISK_PER_TRADE))
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

# LightGBM binary directional mapping (from train.py)
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}


def _enrich_main_ohlcv(repository: HistoricalKlineRepository, symbol: str, main_df: pd.DataFrame) -> pd.DataFrame:
    """Оставлено для совместимости; используйте backtest_data._enrich_main_ohlcv внутри load_all_raw_data."""
    return ld._enrich_main_ohlcv(repository, symbol, main_df)


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


def compute_net_pnl_pct(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    return raw_pnl - (TAKER_COM + TAKER_COM)


def compute_trade_outcome(position: dict, exit_price: float) -> tuple[float, float, float]:
    pnl_clean = compute_net_pnl_pct(position["dir"], position["entry"], exit_price)
    position_notional = float(position["size"])
    commission = position_notional * (TAKER_COM + TAKER_COM)
    trade_profit = position_notional * pnl_clean
    return pnl_clean, trade_profit, commission


def compute_portfolio_equity(balance: float, positions: dict, mark_prices: dict[str, float]) -> float:
    equity = float(balance)
    for sym, position in positions.items():
        if position is None:
            continue
        mark_price = mark_prices.get(sym)
        if mark_price is None or not np.isfinite(mark_price):
            continue
        pnl_clean = compute_net_pnl_pct(position["dir"], position["entry"], float(mark_price))
        equity += float(position["size"]) * pnl_clean
    return equity


def update_drawdown_stats(equity: float, peak_equity: float, max_drawdown: float) -> tuple[float, float]:
    if equity > peak_equity:
        peak_equity = equity
    if peak_equity > 0:
        current_dd = (peak_equity - equity) / peak_equity * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd
    return peak_equity, max_drawdown


def backtest(
    model_name: str = DEFAULT_MODEL_NAME,
    model=None,
    features_meta: dict | None = None,
    predictions: pd.DataFrame | None = None,
    equity_curve_path: Path | None = None,
    result_title: str = "PORTFOLIO BACKTEST RESULTS",
):
    """Делегирует в единый прогон: PortfolioManager + bar_fill + доменный риск/исполнение."""
    from src.application.backtest.portfolio_backtest_runner import run_portfolio_backtest

    return run_portfolio_backtest(
        model_name=model_name,
        model=model,
        features_meta=features_meta,
        predictions=predictions,
        equity_curve_path=equity_curve_path,
        result_title=result_title,
    )


if __name__ == "__main__":
    backtest()
