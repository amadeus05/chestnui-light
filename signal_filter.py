import pandas as pd

import config as cfg

EVENT_FILTER_REQUIRED_COLUMNS = [
    "ema_fast_slow",
    "adx_4h",
    "realized_vol_1h",
]


def _resolve_float_config(*names: str, default: float) -> float:
    for name in names:
        if hasattr(cfg, name):
            return float(getattr(cfg, name))
    return float(default)


def resolve_event_filter_config(override: dict | None = None) -> dict:
    base = {
        "enabled": bool(getattr(cfg, "ENABLE_EVENT_FILTER", False)),
        "min_abs_ema_fast_slow": _resolve_float_config("EVENT_FILTER_MIN_ABS_EMA_FAST_SLOW", default=0.0),
        "min_adx_4h": _resolve_float_config("EVENT_FILTER_MIN_ADX_HTF", "EVENT_FILTER_MIN_ADX_4H", default=0.0),
        "min_realized_vol_1h": _resolve_float_config(
            "EVENT_FILTER_MIN_REALIZED_VOL_MAIN",
            "EVENT_FILTER_MIN_REALIZED_VOL_1H",
            default=0.0,
        ),
        "max_realized_vol_1h": _resolve_float_config(
            "EVENT_FILTER_MAX_REALIZED_VOL_MAIN",
            "EVENT_FILTER_MAX_REALIZED_VOL_1H",
            default=1.0,
        ),
        "required_columns": list(EVENT_FILTER_REQUIRED_COLUMNS),
    }
    if override:
        merged = base.copy()
        merged.update(override)
        merged["required_columns"] = list(override.get("required_columns", EVENT_FILTER_REQUIRED_COLUMNS))
        return merged
    return base


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

    realized_vol = pd.to_numeric(frame["realized_vol_1h"], errors="coerce")
    mask = (
        frame["ema_fast_slow"].abs() >= float(config["min_abs_ema_fast_slow"])
    ) & (
        frame["adx_4h"] >= float(config["min_adx_4h"])
    ) & (
        realized_vol >= float(config["min_realized_vol_1h"])
    ) & (
        realized_vol <= float(config["max_realized_vol_1h"])
    )

    return mask.fillna(False)
