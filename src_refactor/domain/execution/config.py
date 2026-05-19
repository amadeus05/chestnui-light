from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPricingConfig:
    taker_fee: float = 0.0004
    slippage: float = 0.0003


def ensure_pricing_config(config: ExecutionPricingConfig) -> ExecutionPricingConfig:
    if isinstance(config, ExecutionPricingConfig):
        return config
    raise TypeError("Expected ExecutionPricingConfig.")
