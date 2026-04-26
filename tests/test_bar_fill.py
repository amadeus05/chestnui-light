import pytest

from src.domain.execution.bar_fill import (
    NextBarOHLC,
    entry_price_at_bar_open,
    flatten_at_mark_price,
    try_stop_take_fill,
)


def test_long_stop_hit_uses_worse_of_open_and_stop():
    bar = NextBarOHLC(open=100.0, high=101.0, low=98.0)
    slip = 0.001
    r = try_stop_take_fill(
        direction=1,
        entry_price=100.0,
        stop_pct=0.01,
        take_pct=0.02,
        bar=bar,
        slippage=slip,
    )
    assert r is not None
    assert r.reason == "SL"
    stop_level = 100.0 * (1 - 0.01)
    raw = min(bar.open, stop_level)
    assert r.exit_price == pytest.approx(raw * (1 - slip))


def test_short_tp_hit():
    bar = NextBarOHLC(open=100.0, high=100.5, low=97.0)
    slip = 0.0
    entry = 100.0
    take_pct = 0.02
    take_price = entry * (1 - take_pct)
    r = try_stop_take_fill(
        direction=-1,
        entry_price=entry,
        stop_pct=0.01,
        take_pct=take_pct,
        bar=bar,
        slippage=slip,
    )
    assert r is not None
    assert r.reason == "TP"
    assert r.exit_price == pytest.approx(take_price)


def test_no_trigger_when_inside_bar():
    bar = NextBarOHLC(open=100.0, high=100.5, low=99.8)
    r = try_stop_take_fill(
        direction=1,
        entry_price=100.0,
        stop_pct=0.05,
        take_pct=0.10,
        bar=bar,
        slippage=0.0003,
    )
    assert r is None


def test_entry_price_long_short():
    assert entry_price_at_bar_open(1, 100.0, 0.01) == pytest.approx(101.0)
    assert entry_price_at_bar_open(-1, 100.0, 0.01) == pytest.approx(99.0)


def test_flatten_mark():
    assert flatten_at_mark_price(1, 100.0, 0.01) == pytest.approx(99.0)
    assert flatten_at_mark_price(-1, 100.0, 0.01) == pytest.approx(101.0)
