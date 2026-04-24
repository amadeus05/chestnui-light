"""
ETL: логика перенесена из корневого etl.py без изменений формул и порядка шагов.
"""
from .barriers import (
    attach_barrier_columns,
    compute_dynamic_barrier_stop_pct,
    compute_dynamic_barrier_take_pct,
    compute_effective_horizons,
    get_base_horizon,
)
from .candle_maps import build_candle_maps
from .constants import (
    ANSI_RESET,
    ANSI_YELLOW,
    BARRIER_OUTPUT_COLUMNS,
    BASE_OUTPUT_COLUMNS,
    logger,
)
from .contexts import attach_funding_context, attach_open_interest_context, attach_premium_index_context
from .factory import build_labeling_snapshot, create_exchange_service
from .finalize import finalize_feature_frame
from .labeling import (
    compute_clean_pnl,
    resolve_trade_exit,
    simulate_trade_outcome,
    triple_barrier_labeling,
)
from .pipeline import main
from .time_warnings import (
    align_to_next_candle_open,
    format_yellow_warning,
    parse_iso_datetime_to_utc_ms,
    timeframe_to_ms,
    warn_if_history_starts_late,
)

__all__ = [
    "ANSI_RESET",
    "ANSI_YELLOW",
    "BASE_OUTPUT_COLUMNS",
    "BARRIER_OUTPUT_COLUMNS",
    "align_to_next_candle_open",
    "attach_barrier_columns",
    "attach_funding_context",
    "attach_open_interest_context",
    "attach_premium_index_context",
    "build_candle_maps",
    "build_labeling_snapshot",
    "compute_clean_pnl",
    "compute_dynamic_barrier_stop_pct",
    "compute_dynamic_barrier_take_pct",
    "compute_effective_horizons",
    "create_exchange_service",
    "finalize_feature_frame",
    "format_yellow_warning",
    "get_base_horizon",
    "logger",
    "main",
    "parse_iso_datetime_to_utc_ms",
    "resolve_trade_exit",
    "simulate_trade_outcome",
    "timeframe_to_ms",
    "triple_barrier_labeling",
    "warn_if_history_starts_late",
]
