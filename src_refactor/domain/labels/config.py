from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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


def ensure_labeling_config(config: LabelingConfig | Mapping[str, object] | None) -> LabelingConfig:
    if isinstance(config, LabelingConfig):
        return config
    if isinstance(config, Mapping):
        return LabelingConfig.from_mapping(config)
    if config is None:
        return LabelingConfig()
    raise TypeError("Expected LabelingConfig, mapping, or None.")


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
