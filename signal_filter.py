import pandas as pd
import numpy as np

import config as cfg

EVENT_FILTER_REQUIRED_COLUMNS = [
    "ema_fast_slow",
    "adx_4h",
    "realized_vol_1h",
]


# ═══════════════════════════════════════════════════════════════════
# Adaptive Event Filter - 2026-04-05
# Определяет режим волатильности и подстраивает пороги
# ═══════════════════════════════════════════════════════════════════
def detect_volatility_regime(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Определяет режим волатильности для каждого бара: 0=low, 1=normal, 2=high
    Использует rolling percentiles realized_vol_1h
    """
    rvol = pd.to_numeric(frame["realized_vol_1h"], errors="coerce")

    # Параметры из конфига
    low_pctl = float(getattr(cfg, "ADAPTIVE_VOL_PERCENTILE_LOW", 0.25))
    high_pctl = float(getattr(cfg, "ADAPTIVE_VOL_PERCENTILE_HIGH", 0.75))
    lookback = int(getattr(cfg, "ADAPTIVE_VOL_LOOKBACK_BARS", 96))

    # Rolling percentiles волатильности
    vol_low = rvol.rolling(lookback, min_periods=lookback//2).quantile(low_pctl)
    vol_high = rvol.rolling(lookback, min_periods=lookback//2).quantile(high_pctl)

    # Классификация режима
    regime = pd.Series(1, index=index)  # default = normal
    regime[rvol <= vol_low] = 0   # low vol
    regime[rvol >= vol_high] = 2  # high vol

    return regime.fillna(1)


def get_adaptive_thresholds(regime: int) -> dict:
    """Возвращает пороги для конкретного режима волатильности"""
    if regime == 0:  # low vol
        return getattr(cfg, "ADAPTIVE_FILTER_LOW_VOL", {
            "min_abs_ema_fast_slow": 0.003,
            "min_adx_4h": 18.0,
            "min_realized_vol_1h": 0.004,
            "max_realized_vol_1h": 0.015,
        })
    elif regime == 2:  # high vol
        return getattr(cfg, "ADAPTIVE_FILTER_HIGH_VOL", {
            "min_abs_ema_fast_slow": 0.006,
            "min_adx_4h": 24.0,
            "min_realized_vol_1h": 0.020,
            "max_realized_vol_1h": 0.080,
        })
    else:  # normal vol (regime == 1)
        return getattr(cfg, "ADAPTIVE_FILTER_NORMAL_VOL", {
            "min_abs_ema_fast_slow": 0.004,
            "min_adx_4h": 20.0,
            "min_realized_vol_1h": 0.005,
            "max_realized_vol_1h": 0.040,
        })


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

    # ═════════════════════════════════════════════════════════════════
    # Adaptive Event Filter Logic - 2026-04-05
    # ═════════════════════════════════════════════════════════════════
    use_adaptive = bool(getattr(cfg, "ENABLE_ADAPTIVE_EVENT_FILTER", False))

    realized_vol = pd.to_numeric(frame["realized_vol_1h"], errors="coerce")
    ema_fast_slow = pd.to_numeric(frame["ema_fast_slow"], errors="coerce")
    adx_4h = pd.to_numeric(frame["adx_4h"], errors="coerce")

    if use_adaptive:
        # Определяем режим для каждого бара
        regime = detect_volatility_regime(frame, frame.index)

        # Создаем маски для каждого режима
        mask = pd.Series(False, index=frame.index)

        for regime_val in [0, 1, 2]:
            regime_mask = regime == regime_val
            if not regime_mask.any():
                continue

            thresholds = get_adaptive_thresholds(regime_val)

            regime_condition = (
                ema_fast_slow.abs() >= thresholds["min_abs_ema_fast_slow"]
            ) & (
                adx_4h >= thresholds["min_adx_4h"]
            ) & (
                realized_vol >= thresholds["min_realized_vol_1h"]
            ) & (
                realized_vol <= thresholds["max_realized_vol_1h"]
            )

            mask |= (regime_mask & regime_condition)

        # Сохраняем статистику по режимам
        if not hasattr(frame, 'attrs'):
            frame.attrs = {}
        frame.attrs["adaptive_filter_regime_distribution"] = {
            "low_vol": int((regime == 0).sum()),
            "normal_vol": int((regime == 1).sum()),
            "high_vol": int((regime == 2).sum()),
        }
    else:
        # Стандартный (статический) фильтр
        mask = (
            ema_fast_slow.abs() >= float(config["min_abs_ema_fast_slow"])
        ) & (
            adx_4h >= float(config["min_adx_4h"])
        ) & (
            realized_vol >= float(config["min_realized_vol_1h"])
        ) & (
            realized_vol <= float(config["max_realized_vol_1h"])
        )

    return mask.fillna(False)
