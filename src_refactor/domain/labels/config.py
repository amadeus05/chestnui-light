from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LabelingConfig:
    horizon: int = 16
    enable_adaptive_horizon: bool = False
    adaptive_horizon_min: int | None = None
    adaptive_horizon_max: int | None = None
    adaptive_horizon_vol_low: float = 0.005
    adaptive_horizon_vol_high: float = 0.025
    use_dynamic_barriers: bool = True
    barrier_atr_multiplier: float = 1.25
    barrier_rvol_multiplier: float = 0.75
    barrier_tp_to_sl_ratio: float = 2.0
    barrier_min_pct: float | None = None
    barrier_max_pct: float | None = None
    sl_pct: float = 0.015
    tp_pct: float = 0.03
    taker_fee: float = 0.0004
    slippage: float = 0.0003

    @property
    def base_horizon(self) -> int:
        return max(1, int(self.horizon))

    @classmethod
    def from_legacy_config(cls, config: Any) -> "LabelingConfig":
        return cls(
            horizon=int(getattr(config, "HORIZON", 16)),
            enable_adaptive_horizon=bool(getattr(config, "ENABLE_ADAPTIVE_HORIZON", False)),
            adaptive_horizon_min=_optional_int(getattr(config, "ADAPTIVE_HORIZON_MIN", None)),
            adaptive_horizon_max=_optional_int(getattr(config, "ADAPTIVE_HORIZON_MAX", None)),
            adaptive_horizon_vol_low=float(getattr(config, "ADAPTIVE_HORIZON_VOL_LOW", 0.005)),
            adaptive_horizon_vol_high=float(getattr(config, "ADAPTIVE_HORIZON_VOL_HIGH", 0.025)),
            use_dynamic_barriers=bool(getattr(config, "USE_DYNAMIC_BARRIERS", True)),
            barrier_atr_multiplier=float(getattr(config, "BARRIER_ATR_MULTIPLIER", 1.25)),
            barrier_rvol_multiplier=float(getattr(config, "BARRIER_RVOL_MULTIPLIER", 0.75)),
            barrier_tp_to_sl_ratio=float(getattr(config, "BARRIER_TP_TO_SL_RATIO", 2.0)),
            barrier_min_pct=_optional_float(getattr(config, "BARRIER_MIN_PCT", None)),
            barrier_max_pct=_optional_float(getattr(config, "BARRIER_MAX_PCT", None)),
            sl_pct=float(getattr(config, "SL_PCT", 0.015)),
            tp_pct=float(getattr(config, "TP_PCT", 0.03)),
            taker_fee=float(getattr(config, "TAKER_COM", 0.0004)),
            slippage=float(getattr(config, "SLIPPAGE", 0.0003)),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "LabelingConfig":
        if not payload:
            return cls()
        defaults = cls()
        return cls(
            horizon=int(payload.get("horizon", defaults.horizon)),
            enable_adaptive_horizon=bool(payload.get("enable_adaptive_horizon", defaults.enable_adaptive_horizon)),
            adaptive_horizon_min=_optional_int(payload.get("adaptive_horizon_min", defaults.adaptive_horizon_min)),
            adaptive_horizon_max=_optional_int(payload.get("adaptive_horizon_max", defaults.adaptive_horizon_max)),
            adaptive_horizon_vol_low=float(payload.get("adaptive_horizon_vol_low", defaults.adaptive_horizon_vol_low)),
            adaptive_horizon_vol_high=float(payload.get("adaptive_horizon_vol_high", defaults.adaptive_horizon_vol_high)),
            use_dynamic_barriers=bool(payload.get("use_dynamic_barriers", defaults.use_dynamic_barriers)),
            barrier_atr_multiplier=float(payload.get("barrier_atr_multiplier", defaults.barrier_atr_multiplier)),
            barrier_rvol_multiplier=float(payload.get("barrier_rvol_multiplier", defaults.barrier_rvol_multiplier)),
            barrier_tp_to_sl_ratio=float(payload.get("barrier_tp_to_sl_ratio", defaults.barrier_tp_to_sl_ratio)),
            barrier_min_pct=_optional_float(payload.get("barrier_min_pct", defaults.barrier_min_pct)),
            barrier_max_pct=_optional_float(payload.get("barrier_max_pct", defaults.barrier_max_pct)),
            sl_pct=float(payload.get("sl_pct", defaults.sl_pct)),
            tp_pct=float(payload.get("tp_pct", defaults.tp_pct)),
            taker_fee=float(payload.get("taker_fee", defaults.taker_fee)),
            slippage=float(payload.get("slippage", defaults.slippage)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "enable_adaptive_horizon": self.enable_adaptive_horizon,
            "adaptive_horizon_min": self.adaptive_horizon_min,
            "adaptive_horizon_max": self.adaptive_horizon_max,
            "adaptive_horizon_vol_low": self.adaptive_horizon_vol_low,
            "adaptive_horizon_vol_high": self.adaptive_horizon_vol_high,
            "use_dynamic_barriers": self.use_dynamic_barriers,
            "barrier_atr_multiplier": self.barrier_atr_multiplier,
            "barrier_rvol_multiplier": self.barrier_rvol_multiplier,
            "barrier_tp_to_sl_ratio": self.barrier_tp_to_sl_ratio,
            "barrier_min_pct": self.barrier_min_pct,
            "barrier_max_pct": self.barrier_max_pct,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "taker_fee": self.taker_fee,
            "slippage": self.slippage,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.to_mapping(),
            "base_horizon": self.base_horizon,
            "target_column": "Target",
            "barrier_columns": ["barrier_stop_pct", "barrier_take_pct"],
        }


def ensure_labeling_config(config: Any) -> LabelingConfig:
    if isinstance(config, LabelingConfig):
        return config
    if isinstance(config, Mapping):
        return LabelingConfig.from_mapping(config)
    return LabelingConfig.from_legacy_config(config)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
