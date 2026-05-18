from __future__ import annotations

from src_refactor.domain.execution.config import ExecutionPricingConfig, ensure_pricing_config


def compute_raw_pnl_pct(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        return (exit_price - entry_price) / entry_price
    if direction == -1:
        return (entry_price - exit_price) / entry_price
    raise ValueError(f"Unsupported direction: {direction}")


def compute_net_pnl_pct(
    direction: int,
    entry_price: float,
    exit_price: float,
    config: ExecutionPricingConfig,
) -> float:
    pricing = ensure_pricing_config(config)
    return compute_raw_pnl_pct(direction, entry_price, exit_price) - (pricing.taker_fee + pricing.taker_fee)


def compute_fee_quote(notional_quote: float, config: ExecutionPricingConfig) -> float:
    pricing = ensure_pricing_config(config)
    return float(notional_quote) * pricing.taker_fee


def compute_realized_pnl_quote(
    direction: int,
    entry_price: float,
    exit_price: float,
    quantity: float,
) -> float:
    return compute_raw_pnl_pct(direction, entry_price, exit_price) * entry_price * quantity
