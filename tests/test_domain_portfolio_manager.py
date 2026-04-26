"""Тесты PortfolioManager (семантика bt.py)."""
from __future__ import annotations

import pytest

from src.domain.portfolio.models.pnl import compute_portfolio_equity
from src.domain.portfolio.portfolio_manager import PortfolioManager

TAKER = 0.00055


def test_open_and_close_roundtrip_cash():
    pm = PortfolioManager(10_000.0, taker_com=TAKER)
    pm.ensure_symbol_slot("BTC")
    pm.open_position(
        "BTC",
        trade_number=1,
        direction=1,
        entry_price=100.0,
        position_notional=1_000.0,
        required_margin=333.0,
        stop_pct=0.02,
        take_pct=0.04,
        ts_open=None,
    )
    assert pm.used_margin == pytest.approx(333.0)
    assert pm.available_balance() == pytest.approx(10_000.0 - 333.0)

    marks = {"BTC": 100.0}
    eq_before = pm.equity(marks)
    assert eq_before == pytest.approx(
        compute_portfolio_equity(pm.cash, pm.state.positions, marks, taker_com=TAKER)
    )

    pnl_clean, profit, _ = pm.close_position_at_price("BTC", 110.0)
    assert pm.position("BTC") is None
    assert pm.used_margin == 0.0
    raw = (110.0 - 100.0) / 100.0 - 2 * TAKER
    assert pnl_clean == pytest.approx(raw)
    assert profit == pytest.approx(1_000.0 * raw)
    assert pm.cash == pytest.approx(10_000.0 + profit)


def test_observe_equity_updates_drawdown():
    pm = PortfolioManager(100.0, taker_com=TAKER)
    pm.ensure_symbol_slot("ETH")
    pm.open_position(
        "ETH",
        trade_number=1,
        direction=1,
        entry_price=50.0,
        position_notional=80.0,
        required_margin=40.0,
        stop_pct=0.01,
        take_pct=0.02,
        ts_open=None,
    )
    pm.observe_equity({"ETH": 40.0})
    assert pm.peak_equity >= 100.0
    pm.close_position_at_price("ETH", 50.0)
    pm.observe_equity({})
    assert pm.max_drawdown_pct >= 0.0


def test_close_missing_position_raises():
    pm = PortfolioManager(100.0, taker_com=TAKER)
    with pytest.raises(ValueError, match="no open position"):
        pm.close_position_at_price("X", 1.0)
