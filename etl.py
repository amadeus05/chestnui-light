import logging
import re
from datetime import datetime, timezone

import config as cfg
import numpy as np
import pandas as pd

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_service import BinanceService
from src.exchanges.bybit.bybit_service import BybitService
from src.features import MasterFeatureBuilder
from src.features.indicators import compute_atr, safe_ratio
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]


def compute_dynamic_barrier_stop_pct(close: pd.Series, atr_14: pd.Series, realized_vol_1h: pd.Series) -> pd.Series:
    atr_pct = safe_ratio(atr_14, close).abs()
    horizon_vol_pct = realized_vol_1h.abs() * np.sqrt(int(getattr(cfg, "HORIZON", 1)))

    stop_pct = pd.concat(
        [
            atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25)),
            horizon_vol_pct * float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)),
        ],
        axis=1,
    ).max(axis=1)

    min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", getattr(cfg, "SL_PCT", 0.015)))
    max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", getattr(cfg, "TP_PCT", 0.03)))
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
    return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))


def attach_barrier_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)

    if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", True)):
        if "realized_vol_1h" not in output.columns:
            raise ValueError("Dynamic barriers require feature 'realized_vol_1h' to be enabled.")
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(close, atr_14, output["realized_vol_1h"])
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"])
    else:
        output["barrier_stop_pct"] = float(getattr(cfg, "SL_PCT", 0.015))
        output["barrier_take_pct"] = float(getattr(cfg, "TP_PCT", 0.03))
    return output


def compute_clean_pnl(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    taker_com = float(getattr(cfg, "TAKER_COM", 0.0004))
    return raw_pnl - (taker_com + taker_com)


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
) -> tuple[float | None, str | None]:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)

        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - slippage)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - slippage)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)

        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + slippage)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + slippage)
            return exit_price, "TP"

    return None, None


def simulate_trade_outcome(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    stop_pcts: np.ndarray,
    take_pcts: np.ndarray,
    start_idx: int,
    direction: int,
) -> tuple[float, str | None]:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    horizon = int(getattr(cfg, "HORIZON", 16))
    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 + slippage) if direction == 1 else base_open * (1 - slippage)
    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if np.isnan(stop_pct) or np.isnan(take_pct):
        return 0.0, None

    for j in range(1, horizon + 1):
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


def triple_barrier_labeling(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    horizon = int(getattr(cfg, "HORIZON", 16))

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - horizon):
        label = 0
        long_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=1)
        short_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=-1)

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1

        labels.append(label)

    labels.extend([0] * horizon)
    output = df.copy()
    output["Target"] = labels
    return output


def finalize_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    horizon = int(getattr(cfg, "HORIZON", 16))
    if horizon > 0:
        if len(output) <= horizon:
            empty_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
            return output.iloc[0:0][empty_columns].copy()
        output = output.iloc[:-horizon].copy()

    output_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    output.dropna(inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def create_exchange_service() -> ExchangeContract:
    exchange_name = str(getattr(cfg, "ACTIVE_EXCHANGE", "bybit")).strip().lower()
    if exchange_name == "bybit":
        return BybitService()
    if exchange_name == "binance":
        return BinanceService()
    raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {exchange_name}")


def build_labeling_snapshot() -> dict:
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "horizon": int(getattr(cfg, "HORIZON", 0)),
        "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
        "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
        "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
        "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
        "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
        "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
    }


def format_yellow_warning(message: str) -> str:
    return f"{ANSI_YELLOW}{message}{ANSI_RESET}"


def parse_iso_datetime_to_utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def timeframe_to_ms(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip().lower())
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    amount = int(match.group(1))
    unit = match.group(2)
    unit_to_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    return amount * unit_to_ms[unit]


def align_to_next_candle_open(timestamp_ms: int, timeframe: str) -> int:
    timeframe_ms = timeframe_to_ms(timeframe)
    remainder = timestamp_ms % timeframe_ms
    if remainder == 0:
        return timestamp_ms
    return timestamp_ms + (timeframe_ms - remainder)


def warn_if_history_starts_late(
    repository: HistoricalKlineRepository,
    symbol: str,
    timeframe: str,
    requested_start_date: str,
) -> None:
    requested_start_ts = parse_iso_datetime_to_utc_ms(requested_start_date)
    expected_first_open_ts = align_to_next_candle_open(requested_start_ts, timeframe)
    first_open_time = repository.get_first_open_time(symbol, timeframe)
    if first_open_time is None or first_open_time <= expected_first_open_ts:
        return

    logger.warning(
        format_yellow_warning(
            f"[{symbol}-{timeframe}] incomplete history: requested from "
            f"{datetime.fromtimestamp(requested_start_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"expected first candle at "
            f"{datetime.fromtimestamp(expected_first_open_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"but first available candle starts at "
            f"{datetime.fromtimestamp(first_open_time / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}. "
            "The asset was likely listed after the requested start date."
        )
    )


def build_candle_maps(
    repository: HistoricalKlineRepository,
    symbols_to_load: list,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = repository.load_candles(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        htf_df = repository.load_candles(symbol, str(getattr(cfg, "HTF_TIMEFRAME", "4h")))
        funding_df = repository.load_funding_rates(symbol)
        premium_index_df = repository.load_premium_index_klines(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        open_interest_df = repository.load_open_interest(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        if df.empty or htf_df.empty:
            logger.warning("%s: no data in DB (main=%s, htf=%s)", symbol_name, len(df), len(htf_df))
            continue

        df = attach_funding_context(df, funding_df)
        df = attach_premium_index_context(df, premium_index_df)
        df = attach_open_interest_context(df, open_interest_df)
        logger.info(
            "%s: main=%s, htf=%s rows, funding=%s points, premium=%s points, open_interest=%s points",
            symbol_name,
            len(df),
            len(htf_df),
            len(funding_df),
            len(premium_index_df),
            len(open_interest_df),
        )
        base_candle_map[symbol_name] = df
        htf_candle_map[symbol_name] = htf_df

    return base_candle_map, htf_candle_map


def attach_funding_context(
    base_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if funding_df is None or funding_df.empty:
        output["funding_rate"] = np.nan
        return output

    funding_frame = funding_df[["timestamp", "funding_rate"]].copy().sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        output,
        funding_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_premium_index_context(
    base_df: pd.DataFrame,
    premium_index_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if premium_index_df is None or premium_index_df.empty:
        output["premium_index_close"] = np.nan
        return output

    premium_frame = (
        premium_index_df[["timestamp", "premium_index_close"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        premium_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_open_interest_context(
    base_df: pd.DataFrame,
    open_interest_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if open_interest_df is None or open_interest_df.empty:
        output["open_interest"] = np.nan
        return output

    open_interest_frame = (
        open_interest_df[["timestamp", "open_interest"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        open_interest_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def main() -> None:
    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()
    labeling_snapshot = build_labeling_snapshot()

    logger.info(
        "Experiment=%s | labeling_profile=%s | training_profile=%s",
        labeling_snapshot["experiment"],
        labeling_snapshot["labeling_profile"],
        labeling_snapshot["training_profile"],
    )
    logger.info(
        "ETL labeling config: horizon=%s | dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
        labeling_snapshot["horizon"],
        labeling_snapshot["use_dynamic_barriers"],
        labeling_snapshot["barrier_min_pct"],
        labeling_snapshot["barrier_max_pct"],
        labeling_snapshot["barrier_tp_to_sl_ratio"],
    )

    symbols_to_load = [exchange_service.normalize_symbol(symbol) for symbol in dict.fromkeys(getattr(cfg, "SYMBOLS", []))]

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
        htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        start_date = str(getattr(cfg, "START_DATE", "2023-01-01"))
        end_date = getattr(cfg, "END_DATE", None)

        logger.info("Loading %s %s from %s...", symbol_name, timeframe, start_date)
        loaded = repository.sync_candles(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, timeframe, loaded)
        warn_if_history_starts_late(repository, symbol_name, timeframe, start_date)

        logger.info("Loading %s %s from %s...", symbol_name, htf_timeframe, start_date)
        htf_loaded = repository.sync_candles(exchange_service, symbol, htf_timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, htf_timeframe, htf_loaded)
        warn_if_history_starts_late(repository, symbol_name, htf_timeframe, start_date)

        logger.info("Loading %s funding from %s...", symbol_name, start_date)
        funding_loaded = repository.sync_funding_rates(exchange_service, symbol, start_date, end_date)
        logger.info("%s funding: %s new points", symbol_name, funding_loaded)

        logger.info("Loading %s premium index %s from %s...", symbol_name, timeframe, start_date)
        premium_loaded = repository.sync_premium_index_klines(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s premium index %s: %s new candles", symbol_name, timeframe, premium_loaded)

        logger.info("Loading %s open interest %s from %s...", symbol_name, timeframe, start_date)
        open_interest_loaded = repository.sync_open_interest(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s open interest %s: %s new points", symbol_name, timeframe, open_interest_loaded)

    base_candle_map, htf_candle_map = build_candle_maps(repository, symbols_to_load)
    feature_builder = MasterFeatureBuilder()
    pipeline_result = feature_builder.build(base_candle_map, htf_candle_map)
    logger.info(
        "Feature build request resolved: profile=%s | blocks=%s | features=%s",
        pipeline_result.profile_name,
        ", ".join(pipeline_result.active_blocks) or "none",
        len(pipeline_result.feature_columns),
    )

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        feature_df = pipeline_result.feature_map.get(symbol_name)
        if feature_df is None or feature_df.empty:
            logger.warning("%s: skipped, missing prepared feature inputs", symbol_name)
            continue

        feature_df = attach_barrier_columns(feature_df)
        feature_df = triple_barrier_labeling(feature_df)
        feature_df = finalize_feature_frame(feature_df, list(pipeline_result.feature_columns))
        repository.save_features(symbol, feature_df)
        logger.info("%s: saved %s rows with %s requested features", symbol_name, len(feature_df), len(pipeline_result.feature_columns))


if __name__ == "__main__":
    main()
