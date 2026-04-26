"""
Единая модель исполнения по следующей свече (OHLC + slippage), как в bt.py / backtest_engine.
Бэктест, paper и тесты вызывают один и тот же код; live после факта использует домен, а не этот слой.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NextBarOHLC:
    open: float
    high: float
    low: float


@dataclass(frozen=True)
class StopTakeFillResult:
    reason: str  # "TP" | "SL"
    exit_price: float


def try_stop_take_fill(
    direction: int,
    entry_price: float,
    stop_pct: float,
    take_pct: float,
    bar: NextBarOHLC,
    slippage: float,
) -> StopTakeFillResult | None:
    """
    Проверка TP/SL на баре после входа (как в бэктесте: фаза выходов по next_open/high/low).
    Возвращает результат, если сработал стоп или тейк; иначе None.
    """
    next_open = float(bar.open)
    next_high = float(bar.high)
    next_low = float(bar.low)
    ep = float(entry_price)
    sp = float(stop_pct)
    tp = float(take_pct)
    slip = float(slippage)

    if direction == 1:
        stop_price = ep * (1 - sp)
        take_price = ep * (1 + tp)
        if next_low <= stop_price:
            raw = next_open if next_open < stop_price else stop_price
            return StopTakeFillResult(reason="SL", exit_price=raw * (1 - slip))
        if next_high >= take_price:
            return StopTakeFillResult(reason="TP", exit_price=take_price * (1 - slip))
        return None

    if direction == -1:
        stop_price = ep * (1 + sp)
        take_price = ep * (1 - tp)
        if next_high >= stop_price:
            raw = next_open if next_open > stop_price else stop_price
            return StopTakeFillResult(reason="SL", exit_price=raw * (1 + slip))
        if next_low <= take_price:
            return StopTakeFillResult(reason="TP", exit_price=take_price * (1 + slip))
        return None

    raise ValueError(f"direction must be 1 or -1, got {direction}")


def entry_price_at_bar_open(direction: int, next_open: float, slippage: float) -> float:
    """Цена входа по открытию следующей свечи с учётом slippage (лонг / шорт)."""
    o = float(next_open)
    s = float(slippage)
    if direction == 1:
        return o * (1 + s)
    if direction == -1:
        return o * (1 - s)
    raise ValueError(f"direction must be 1 or -1, got {direction}")


def flatten_at_mark_price(direction: int, mark_price: float, slippage: float) -> float:
    """Закрытие по последней отметке (конец бэктеста): mark ± slippage."""
    m = float(mark_price)
    s = float(slippage)
    if direction == 1:
        return m * (1 - s)
    if direction == -1:
        return m * (1 + s)
    raise ValueError(f"direction must be 1 or -1, got {direction}")
