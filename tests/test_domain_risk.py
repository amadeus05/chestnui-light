"""Тесты RiskManager (effective risk + сайзинг как в bt.py)."""
from __future__ import annotations

import pytest

from src.domain.risk import RiskLimits, RiskManager


def test_effective_risk_unreduced_until_threshold():
    limits = RiskLimits(
        risk_per_trade=0.02,
        reduced_risk_per_trade=0.005,
        reduce_risk_after_consecutive_losses=4,
        leverage=5.0,
    )
    rm = RiskManager(limits)
    for k in range(4):
        assert rm.effective_risk_per_trade(k) == pytest.approx(0.02)
    assert rm.effective_risk_per_trade(4) == pytest.approx(0.005)
    assert rm.effective_risk_per_trade(99) == pytest.approx(0.005)


def test_effective_risk_no_reduce_when_disabled():
    limits = RiskLimits(
        risk_per_trade=0.02,
        reduced_risk_per_trade=0.005,
        reduce_risk_after_consecutive_losses=0,
        leverage=5.0,
    )
    rm = RiskManager(limits)
    assert rm.effective_risk_per_trade(100) == pytest.approx(0.02)


def test_effective_risk_no_reduce_when_reduced_not_lower():
    limits = RiskLimits(
        risk_per_trade=0.02,
        reduced_risk_per_trade=0.03,
        reduce_risk_after_consecutive_losses=2,
        leverage=5.0,
    )
    rm = RiskManager(limits)
    assert rm.effective_risk_per_trade(5) == pytest.approx(0.02)


def test_raw_entry_notional_and_margin_matches_bt_formula():
    limits = RiskLimits(
        risk_per_trade=0.01,
        reduced_risk_per_trade=0.01,
        reduce_risk_after_consecutive_losses=0,
        leverage=3.0,
        min_position_notional=10.0,
    )
    rm = RiskManager(limits)
    snap = 1000.0
    stop = 0.02
    n, m = rm.raw_entry_notional_and_margin(snap, stop, 0)
    risk_capital = snap * 0.01
    expect_n = min(risk_capital / stop, snap * 3.0)
    assert n == pytest.approx(expect_n)
    assert m == pytest.approx(expect_n / 3.0)


def test_passes_min_notional():
    limits = RiskLimits(0.01, 0.01, 0, 2.0, min_position_notional=10.0)
    rm = RiskManager(limits)
    assert rm.passes_min_notional(10.0)
    assert not rm.passes_min_notional(9.99)
