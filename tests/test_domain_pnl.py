"""Тесты формул из bt.py: src.domain.portfolio.models.pnl."""
from __future__ import annotations

import pytest

from src.domain.portfolio.models.pnl import (
    compute_net_pnl_pct,
    compute_portfolio_equity,
    compute_trade_outcome,
    update_drawdown_stats,
)

TAKER = 0.0004


def _net_long(entry: float, exit_px: float) -> float:
    return (exit_px - entry) / entry - 2 * TAKER


def _net_short(entry: float, exit_px: float) -> float:
    return (entry - exit_px) / entry - 2 * TAKER


def test_compute_net_pnl_pct_long():
    assert compute_net_pnl_pct(1, 100.0, 100.0, taker_com=TAKER) == pytest.approx(-2 * TAKER)
    assert compute_net_pnl_pct(1, 100.0, 101.0, taker_com=TAKER) == pytest.approx(_net_long(100.0, 101.0))


def test_compute_net_pnl_pct_short():
    assert compute_net_pnl_pct(-1, 100.0, 99.0, taker_com=TAKER) == pytest.approx(_net_short(100.0, 99.0))


def test_compute_trade_outcome_matches_manual():
    pos = {"dir": 1, "entry": 200.0, "size": 300.0}
    exit_px = 204.0
    pnl_clean, trade_profit, commission = compute_trade_outcome(pos, exit_px, taker_com=TAKER)
    assert pnl_clean == pytest.approx(_net_long(200.0, 204.0))
    assert commission == pytest.approx(300.0 * 2 * TAKER)
    assert trade_profit == pytest.approx(300.0 * pnl_clean)


def test_compute_portfolio_equity_cash_only():
    eq = compute_portfolio_equity(1_000.0, {}, {}, taker_com=TAKER)
    assert eq == 1_000.0


def test_compute_portfolio_equity_skips_none_and_bad_marks():
    positions = {
        "A": None,
        "B": {"dir": 1, "entry": 100.0, "size": 50.0},
    }
    marks = {"B": 100.0}
    eq = compute_portfolio_equity(900.0, positions, marks, taker_com=TAKER)
    pnl_B = 50.0 * _net_long(100.0, 100.0)
    assert eq == pytest.approx(900.0 + pnl_B)

    marks_bad = {"B": float("nan")}
    eq2 = compute_portfolio_equity(900.0, positions, marks_bad, taker_com=TAKER)
    assert eq2 == 900.0

    marks_inf = {"B": float("inf")}
    eq3 = compute_portfolio_equity(900.0, positions, marks_inf, taker_com=TAKER)
    assert eq3 == 900.0


def test_compute_portfolio_equity_two_positions():
    positions = {
        "X": {"dir": 1, "entry": 100.0, "size": 10.0},
        "Y": {"dir": -1, "entry": 50.0, "size": 20.0},
    }
    marks = {"X": 102.0, "Y": 49.0}
    eq = compute_portfolio_equity(500.0, positions, marks, taker_com=TAKER)
    add_x = 10.0 * _net_long(100.0, 102.0)
    add_y = 20.0 * _net_short(50.0, 49.0)
    assert eq == pytest.approx(500.0 + add_x + add_y)


def test_update_drawdown_stats_monotone_max():
    peak, mx = 100.0, 0.0
    peak, mx = update_drawdown_stats(120.0, peak, mx)
    assert peak == 120.0 and mx == 0.0
    peak, mx = update_drawdown_stats(90.0, peak, mx)
    assert peak == 120.0
    assert mx == pytest.approx((120.0 - 90.0) / 120.0 * 100)
    peak, mx = update_drawdown_stats(60.0, peak, mx)
    assert mx == pytest.approx((120.0 - 60.0) / 120.0 * 100)
    peak, mx = update_drawdown_stats(130.0, peak, mx)
    assert peak == 130.0
    assert mx == pytest.approx(50.0)


def test_update_drawdown_stats_zero_peak_branch():
    peak, mx = 0.0, 0.0
    peak, mx = update_drawdown_stats(-10.0, peak, mx)
    assert peak == 0.0
    assert mx == 0.0
