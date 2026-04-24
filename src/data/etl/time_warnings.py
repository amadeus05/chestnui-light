import re
from datetime import datetime, timezone

from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

from .constants import ANSI_RESET, ANSI_YELLOW, logger


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
