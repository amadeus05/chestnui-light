import pandas as pd

import config as cfg

EVENT_FILTER_REQUIRED_COLUMNS = [
    "ema_fast_slow",
    "adx_4h",
    "realized_vol_1h",
]


def _cfg_bool(name: str, default: bool = False) -> bool:
    return bool(getattr(cfg, name, default))


def _cfg_float(*names: str, default: float = 0.0) -> float:
    for name in names:
        if hasattr(cfg, name):
            return float(getattr(cfg, name))
    return float(default)


def _cfg_str(*names: str, default: str = "") -> str:
    for name in names:
        if hasattr(cfg, name):
            return str(getattr(cfg, name))
    return str(default)


def resolve_event_filter_config(override: dict | None = None) -> dict:
    min_ema_fast_slow = _cfg_float(
        "EVENT_FILTER_MIN_EMA_FAST_SLOW",
        "EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW",
        default=0.0,
    )
    base = {
        "enabled": _cfg_bool("ENABLE_EVENT_FILTER", False),
        "side": _cfg_str("EVENT_FILTER_SIDE", default="both").strip().lower(),
        "min_ema_fast_slow": min_ema_fast_slow,
        "min_abs_ema_fast_slow": min_ema_fast_slow,
        "min_adx_4h": _cfg_float("EVENT_FILTER_MIN_ADX_4H", "EVENT_FILTER_MIN_ADX_HTF", default=0.0),
        "min_realized_vol_1h": _cfg_float(
            "EVENT_FILTER_MIN_REALIZED_VOL_1H",
            "EVENT_FILTER_MIN_REALIZED_VOL_MAIN",
            default=0.0,
        ),
        "max_realized_vol_1h": _cfg_float(
            "EVENT_FILTER_MAX_REALIZED_VOL_1H",
            "EVENT_FILTER_MAX_REALIZED_VOL_MAIN",
            default=1.0,
        ),
        "required_columns": list(EVENT_FILTER_REQUIRED_COLUMNS),
    }
    if override:
        merged = base.copy()
        merged.update(override)
        merged["side"] = str(merged.get("side", "both")).strip().lower()
        merged["required_columns"] = list(override.get("required_columns", EVENT_FILTER_REQUIRED_COLUMNS))
        return merged
    return base


def resolve_backtest_event_filter_config(base_config: dict | None = None, override: dict | None = None) -> dict:
    base = resolve_event_filter_config(base_config)
    min_ema_fast_slow = _cfg_float(
        "BACKTEST_EVENT_FILTER_MIN_EMA_FAST_SLOW",
        "BACKTEST_EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW",
        default=float(base.get("min_ema_fast_slow", base.get("min_abs_ema_fast_slow", 0.0))),
    )
    resolved = {
        "enabled": _cfg_bool("BACKTEST_ENABLE_EVENT_FILTER", bool(base.get("enabled", False))),
        "side": _cfg_str("BACKTEST_EVENT_FILTER_SIDE", default=str(base.get("side", "both"))).strip().lower(),
        "min_ema_fast_slow": min_ema_fast_slow,
        "min_abs_ema_fast_slow": min_ema_fast_slow,
        "min_adx_4h": _cfg_float(
            "BACKTEST_EVENT_FILTER_MIN_ADX_4H",
            "BACKTEST_EVENT_FILTER_MIN_ADX_HTF",
            default=float(base.get("min_adx_4h", 0.0)),
        ),
        "min_realized_vol_1h": _cfg_float(
            "BACKTEST_EVENT_FILTER_MIN_REALIZED_VOL_1H",
            "BACKTEST_EVENT_FILTER_MIN_REALIZED_VOL_MAIN",
            default=float(base.get("min_realized_vol_1h", 0.0)),
        ),
        "max_realized_vol_1h": _cfg_float(
            "BACKTEST_EVENT_FILTER_MAX_REALIZED_VOL_1H",
            "BACKTEST_EVENT_FILTER_MAX_REALIZED_VOL_MAIN",
            default=float(base.get("max_realized_vol_1h", 1.0)),
        ),
        "required_columns": list(base.get("required_columns", EVENT_FILTER_REQUIRED_COLUMNS)),
    }
    if override:
        merged = resolved.copy()
        merged.update(override)
        merged["side"] = str(merged.get("side", "both")).strip().lower()
        merged["required_columns"] = list(override.get("required_columns", resolved["required_columns"]))
        return merged
    return resolved


def validate_event_filter_columns(frame: pd.DataFrame, event_filter_config: dict | None = None) -> None:
    config = resolve_event_filter_config(event_filter_config)
    if not config.get("enabled", False):
        return

    missing = [column for column in config["required_columns"] if column not in frame.columns]
    if missing:
        raise ValueError(
            "Event filter requires missing columns: " + ", ".join(missing)
        )


def build_candidate_event_mask(frame: pd.DataFrame, event_filter_config: dict | None = None) -> pd.Series:
    config = resolve_event_filter_config(event_filter_config)
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    if not config.get("enabled", False):
        return pd.Series(True, index=frame.index)

    validate_event_filter_columns(frame, config)

    ema_fast_slow = pd.to_numeric(frame["ema_fast_slow"], errors="coerce")
    realized_vol = pd.to_numeric(frame["realized_vol_1h"], errors="coerce")

    side = str(config.get("side", "both")).strip().lower()
    trend_threshold = float(config.get("min_ema_fast_slow", config.get("min_abs_ema_fast_slow", 0.0)))
    if side == "long":
        trend_mask = ema_fast_slow >= trend_threshold
    elif side == "short":
        trend_mask = ema_fast_slow <= -trend_threshold
    else:
        trend_mask = ema_fast_slow.abs() >= float(config.get("min_abs_ema_fast_slow", trend_threshold))

    mask = trend_mask & (
        frame["adx_4h"] >= float(config["min_adx_4h"])
    ) & (
        realized_vol >= float(config["min_realized_vol_1h"])
    ) & (
        realized_vol <= float(config["max_realized_vol_1h"])
    )

    return mask.fillna(False)
