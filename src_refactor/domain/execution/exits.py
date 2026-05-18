from __future__ import annotations

from dataclasses import dataclass

from src_refactor.domain.execution.config import ExecutionPricingConfig, ensure_pricing_config


@dataclass(frozen=True, slots=True)
class TradeExit:
    price: float | None
    reason: str | None

    @property
    def exit_price(self) -> float | None:
        return self.price


def apply_entry_slippage(direction: int, base_open: float, config: ExecutionPricingConfig) -> float:
    if direction == 1:
        return base_open * (1 + config.slippage)
    if direction == -1:
        return base_open * (1 - config.slippage)
    raise ValueError(f"Unsupported direction: {direction}")


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
    config: ExecutionPricingConfig,
) -> TradeExit:
    pricing = ensure_pricing_config(config)
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)
        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - pricing.slippage)
            return TradeExit(price=exit_price, reason="SL")
        if next_high >= take_price:
            exit_price = take_price * (1 - pricing.slippage)
            return TradeExit(price=exit_price, reason="TP")
        return TradeExit(price=None, reason=None)

    if direction == -1:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)
        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + pricing.slippage)
            return TradeExit(price=exit_price, reason="SL")
        if next_low <= take_price:
            exit_price = take_price * (1 + pricing.slippage)
            return TradeExit(price=exit_price, reason="TP")
        return TradeExit(price=None, reason=None)

    raise ValueError(f"Unsupported direction: {direction}")
