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
from src.features.indicators import compute_atr, compute_donchian_channels, safe_ratio
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
    realized_vol_1h: pd.Series | None = None,
) -> pd.Series:
    atr_pct = safe_ratio(atr_14, close).abs()
    barrier_candidates = [atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25))]
    if realized_vol_1h is not None:
        horizon_vol_pct = realized_vol_1h.abs() * np.sqrt(int(getattr(cfg, "HORIZON", 1)))
        barrier_candidates.append(horizon_vol_pct * float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)))

    stop_pct = pd.concat(barrier_candidates, axis=1).max(axis=1)

    min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", getattr(cfg, "SL_PCT", 0.015)))
    max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", getattr(cfg, "TP_PCT", 0.03)))
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
    return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))


def resolve_barrier_mode() -> str:
    mode = str(getattr(cfg, "BARRIER_MODE", "")).strip().lower()
    if mode:
        return mode
    if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", True)):
        return "dynamic"
    return "fixed"


def compute_barrier_realized_vol(close: pd.Series) -> pd.Series:
    timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
    bars_per_hour = max(1, int(round(3_600_000 / timeframe_to_ms(timeframe))))
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(bars_per_hour, min_periods=bars_per_hour).std()


def compute_strategy_barrier_pcts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = df["close"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]

    donchian_length = int(getattr(cfg, "DONCHIAN_LENGTH", 96))
    local_extreme_lookback = int(getattr(cfg, "STRATEGY_BARRIER_LOCAL_EXTREME_LOOKBACK", 12))
    max_midline_pct = float(getattr(cfg, "STRATEGY_BARRIER_MAX_MIDLINE_PCT", 0.03))
    min_stop_pct = float(getattr(cfg, "STRATEGY_BARRIER_MIN_PCT", 0.0))
    tp_to_sl_ratio = float(getattr(cfg, "STRATEGY_BARRIER_TP_TO_SL_RATIO", 2.0))

    _, donchian_mid, _ = compute_donchian_channels(high, low, length=donchian_length, shift=1)
    local_swing_low = low.rolling(local_extreme_lookback, min_periods=local_extreme_lookback).min().shift(1)

    midline_stop_pct = safe_ratio((close - donchian_mid).clip(lower=0.0), close)
    local_low_stop_pct = safe_ratio((close - local_swing_low).clip(lower=0.0), close)

    use_local_extreme = (
        donchian_mid.isna()
        | (donchian_mid >= close)
        | (midline_stop_pct > max_midline_pct)
    )
    stop_pct = midline_stop_pct.where(~use_local_extreme, local_low_stop_pct)
    stop_pct = stop_pct.where(stop_pct.notna(), midline_stop_pct)
    stop_pct = stop_pct.where(stop_pct > 0.0)
    stop_pct = stop_pct.clip(lower=min_stop_pct)
    take_pct = stop_pct * tp_to_sl_ratio
    return stop_pct, take_pct


def attach_barrier_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)
    barrier_mode = resolve_barrier_mode()

    if barrier_mode == "dynamic":
        realized_vol = (
            output["realized_vol_1h"]
            if "realized_vol_1h" in output.columns
            else compute_barrier_realized_vol(close)
        )
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(close, atr_14, realized_vol)
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"])
    elif barrier_mode == "donchian_midline_rr":
        output["barrier_stop_pct"], output["barrier_take_pct"] = compute_strategy_barrier_pcts(output)
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


def resolve_vertical_barrier_exit(direction: int, final_close: float) -> float:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    if direction == 1:
        return final_close * (1 - slippage)
    return final_close * (1 + slippage)


def compute_label_min_net_return_thresholds(df: pd.DataFrame) -> np.ndarray:
    """Порог net-доходности по бару: gross > avg_spread + 2*TAKER_COM ⇔ long_pnl > avg_spread."""
    n = len(df)
    floor = float(getattr(cfg, "LABEL_AVERAGE_SPREAD_PCT", 0.0))
    if not bool(getattr(cfg, "LABEL_USE_ROLLING_SPREAD_PROXY", True)):
        return np.full(n, floor, dtype=np.float64)
    window = int(getattr(cfg, "LABEL_SPREAD_ROLLING_BARS", 24))
    close = df["close"].replace(0, np.nan)
    hl_range = (df["high"] - df["low"]) / close
    min_periods = max(1, min(window, window // 2 or 1))
    roll = hl_range.rolling(window, min_periods=min_periods).mean()
    return np.maximum(np.nan_to_num(roll.values.astype(np.float64), nan=floor), floor)


def resolve_long_label(
    long_pnl: float,
    exit_reason: str | None,
    min_net_threshold: float = 0.0,
) -> float:
    use_sig = bool(getattr(cfg, "LABEL_USE_SIGNIFICANT_RETURN", False))

    if exit_reason == "TIME":
        neutral_band = float(getattr(cfg, "TIME_EXIT_NEUTRAL_BAND", 0.0))
        if use_sig:
            if abs(long_pnl) <= neutral_band:
                return np.nan
            if long_pnl < -neutral_band:
                return 0.0
            return float(long_pnl > min_net_threshold)
        if long_pnl > neutral_band:
            return 1.0
        if long_pnl < -neutral_band:
            return 0.0
        return np.nan

    if use_sig:
        return float(long_pnl > min_net_threshold)
    return float(long_pnl > 0)


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
    closes: np.ndarray,
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

    final_candle_idx = min(start_idx + horizon, len(closes) - 1)
    final_exit_price = resolve_vertical_barrier_exit(direction, closes[final_candle_idx])
    return compute_clean_pnl(direction, entry_price, final_exit_price), "TIME"


def triple_barrier_labeling(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    horizon = int(getattr(cfg, "HORIZON", 16))

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values
    min_net_thresholds = compute_label_min_net_return_thresholds(df)

    for i in range(len(df) - horizon):
        long_pnl, exit_reason = simulate_trade_outcome(opens, highs, lows, closes, stop_pcts, take_pcts, i, direction=1)
        labels.append(resolve_long_label(long_pnl, exit_reason, float(min_net_thresholds[i])))

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
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = repository.load_candles(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        htf_df = repository.load_candles(symbol, str(getattr(cfg, "HTF_TIMEFRAME", "4h")))
        if df.empty or htf_df.empty:
            logger.warning("%s: no data in DB (main=%s, htf=%s)", symbol_name, len(df), len(htf_df))
            continue

        logger.info("%s: main=%s, htf=%s rows", symbol_name, len(df), len(htf_df))
        base_candle_map[symbol_name] = df
        htf_candle_map[symbol_name] = htf_df

    return (base_candle_map, htf_candle_map)


def sync_bybit_market_context(
    repository: HistoricalKlineRepository,
    exchange_service: BybitService,
    symbol: str,
    start_date: str,
    end_date: str | None,
) -> None:
    for timeframe in (
        str(getattr(cfg, "TIMEFRAME", "1h")),
        str(getattr(cfg, "HTF_TIMEFRAME", "4h")),
    ):
        logger.info("Loading %s %s open-interest from %s...", symbol, timeframe, start_date)
        open_interest_loaded = repository.sync_open_interest(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s %s open-interest: %s new rows", symbol, timeframe, open_interest_loaded)

    logger.info("Loading %s funding history from %s...", symbol, start_date)
    funding_loaded = repository.sync_funding_rates(exchange_service, symbol, start_date, end_date)
    logger.info("%s funding history: %s new rows", symbol, funding_loaded)


def main() -> None:
    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()

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

        if isinstance(exchange_service, BybitService):
            sync_bybit_market_context(
                repository=repository,
                exchange_service=exchange_service,
                symbol=symbol_name,
                start_date=start_date,
                end_date=end_date,
            )

    (base_candle_map, htf_candle_map) = build_candle_maps(repository, symbols_to_load)
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
