from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionPricingConfig:
    taker_fee: float = 0.0004
    slippage: float = 0.0003

    @classmethod
    def from_legacy_config(cls, config: Any) -> "ExecutionPricingConfig":
        return cls(
            taker_fee=float(getattr(config, "TAKER_COM", 0.0004)),
            slippage=float(getattr(config, "SLIPPAGE", 0.0003)),
        )


def ensure_pricing_config(config: ExecutionPricingConfig | Any) -> ExecutionPricingConfig:
    if isinstance(config, ExecutionPricingConfig):
        return config
    return ExecutionPricingConfig.from_legacy_config(config)
