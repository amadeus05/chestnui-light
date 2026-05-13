import math

import numpy as np
import pytest

import bt
import config as cfg
import etl


@pytest.fixture()
def execution_config(monkeypatch):
    monkeypatch.setattr(cfg, "SLIPPAGE", 0.001)
    monkeypatch.setattr(cfg, "TAKER_COM", 0.0004)
    monkeypatch.setattr(bt, "SLIPPAGE", 0.001)
    monkeypatch.setattr(bt, "TAKER_COM", 0.0004)
    monkeypatch.setattr(bt, "DIRECTIONAL_PROBA_THRESHOLD", 0.55)
    monkeypatch.setattr(bt, "MIN_SIGNAL_GAP", 0.02)


@pytest.mark.parametrize(
    ("direction", "entry", "next_open", "next_high", "next_low", "stop_pct", "take_pct", "expected_price", "expected_reason"),
    [
        (1, 100.0, 100.0, 104.0, 99.0, 0.02, 0.03, 103.0 * 0.999, "TP"),
        (1, 100.0, 100.0, 101.0, 97.0, 0.02, 0.03, 98.0 * 0.999, "SL"),
        (1, 100.0, 96.0, 101.0, 95.0, 0.02, 0.03, 96.0 * 0.999, "SL"),
        (-1, 100.0, 100.0, 101.0, 96.0, 0.02, 0.03, 97.0 * 1.001, "TP"),
        (-1, 100.0, 100.0, 103.0, 99.0, 0.02, 0.03, 102.0 * 1.001, "SL"),
        (-1, 100.0, 104.0, 105.0, 99.0, 0.02, 0.03, 104.0 * 1.001, "SL"),
        (1, 100.0, 100.0, 102.0, 99.0, 0.02, 0.03, None, None),
        (-1, 100.0, 100.0, 101.0, 98.0, 0.02, 0.03, None, None),
    ],
)
def test_resolve_trade_exit_behavior_and_etl_bt_parity(
    execution_config,
    direction,
    entry,
    next_open,
    next_high,
    next_low,
    stop_pct,
    take_pct,
    expected_price,
    expected_reason,
):
    etl_exit = etl.resolve_trade_exit(
        direction,
        entry,
        next_open,
        next_high,
        next_low,
        stop_pct,
        take_pct,
    )
    bt_exit = bt.resolve_trade_exit(
        direction,
        entry,
        next_open,
        next_high,
        next_low,
        stop_pct,
        take_pct,
    )

    assert etl_exit == bt_exit
    exit_price, reason = etl_exit
    assert reason == expected_reason
    if expected_price is None:
        assert exit_price is None
    else:
        assert exit_price == pytest.approx(expected_price)


def test_resolve_trade_exit_checks_stop_before_take_when_both_hit(execution_config):
    exit_price, reason = etl.resolve_trade_exit(
        direction=1,
        entry_price=100.0,
        next_open=100.0,
        next_high=104.0,
        next_low=97.0,
        stop_pct=0.02,
        take_pct=0.03,
    )

    assert reason == "SL"
    assert exit_price == pytest.approx(98.0 * 0.999)


def test_entry_candidate_exit_uses_candidate_symbol_market_snapshot(execution_config):
    candidate = {
        "sym": "BTC/USDT",
        "signal": 1,
        "entry_price": 30_000.0,
        "stop_pct": 0.01,
        "take_pct": 0.02,
    }
    market_batch = {
        "SOL/USDT": {
            "next_open": 25.0,
            "next_high": 26.0,
            "next_low": 24.0,
        },
        "BTC/USDT": {
            "next_open": 30_000.0,
            "next_high": 30_100.0,
            "next_low": 29_500.0,
        },
    }

    exit_price, reason = bt.resolve_entry_candidate_exit(candidate, market_batch)

    assert reason == "SL"
    assert exit_price == pytest.approx(29_700.0 * 0.999)
    assert bt.compute_net_pnl_pct(1, candidate["entry_price"], exit_price) > -0.02


@pytest.mark.parametrize(
    ("direction", "entry", "exit_price", "expected_raw"),
    [
        (1, 100.0, 110.0, 0.10),
        (1, 100.0, 90.0, -0.10),
        (-1, 100.0, 90.0, 0.10),
        (-1, 100.0, 110.0, -0.10),
    ],
)
def test_pnl_formula_behavior_and_etl_bt_parity(
    execution_config,
    direction,
    entry,
    exit_price,
    expected_raw,
):
    expected_net = expected_raw - 0.0008

    assert etl.compute_clean_pnl(direction, entry, exit_price) == pytest.approx(expected_net)
    assert bt.compute_net_pnl_pct(direction, entry, exit_price) == pytest.approx(expected_net)
    assert etl.compute_clean_pnl(direction, entry, exit_price) == pytest.approx(
        bt.compute_net_pnl_pct(direction, entry, exit_price)
    )


def test_compute_trade_outcome_uses_notional_and_commission(execution_config):
    position = {"dir": 1, "entry": 100.0, "size": 250.0}

    pnl_clean, trade_profit, commission = bt.compute_trade_outcome(position, exit_price=110.0)

    assert pnl_clean == pytest.approx(0.10 - 0.0008)
    assert trade_profit == pytest.approx(250.0 * (0.10 - 0.0008))
    assert commission == pytest.approx(250.0 * 0.0008)


def test_simulate_trade_outcome_uses_next_open_as_slipped_entry(execution_config):
    opens = np.array([100.0, 101.0, 101.0, 101.0])
    highs = np.array([100.0, 104.5, 101.0, 101.0])
    lows = np.array([100.0, 100.0, 100.0, 100.0])
    stop_pcts = np.array([0.02, 0.02, 0.02, 0.02])
    take_pcts = np.array([0.03, 0.03, 0.03, 0.03])

    pnl, reason = etl.simulate_trade_outcome(
        opens,
        highs,
        lows,
        stop_pcts,
        take_pcts,
        start_idx=0,
        direction=1,
        horizon=2,
    )

    entry = 101.0 * 1.001
    exit_price = (entry * 1.03) * 0.999
    expected_pnl = (exit_price - entry) / entry - 0.0008
    assert reason == "TP"
    assert pnl == pytest.approx(expected_pnl)


def test_simulate_trade_outcome_returns_neutral_when_barriers_are_nan(execution_config):
    values = np.array([100.0, 101.0, 102.0])

    pnl, reason = etl.simulate_trade_outcome(
        opens=values,
        highs=values + 10.0,
        lows=values - 10.0,
        stop_pcts=np.array([math.nan, 0.02, 0.02]),
        take_pcts=np.array([0.03, 0.03, 0.03]),
        start_idx=0,
        direction=1,
        horizon=2,
    )

    assert pnl == 0.0
    assert reason is None


@pytest.mark.parametrize(
    ("p_long", "p_short", "expected_direction", "expected_probability", "expected_gap"),
    [
        (0.58, 0.54, 1, 0.58, 0.04),
        (0.52, 0.57, -1, 0.57, 0.05),
        (0.56, 0.55, 0, 0.56, 0.01),
        (0.54, 0.20, 0, 0.54, 0.34),
    ],
)
def test_resolve_directional_signal_threshold_and_gap(
    execution_config,
    p_long,
    p_short,
    expected_direction,
    expected_probability,
    expected_gap,
):
    direction, probability, gap = bt.resolve_directional_signal(p_long, p_short)

    assert direction == expected_direction
    assert probability == pytest.approx(expected_probability)
    assert gap == pytest.approx(expected_gap)


def test_build_entry_score_uses_edge_above_threshold_and_gap(execution_config):
    assert bt.build_entry_score(direction_prob=0.60, signal_gap=0.03) == pytest.approx(0.53)
    assert bt.build_entry_score(direction_prob=0.50, signal_gap=0.03) == pytest.approx(0.03)


def test_get_barrier_pcts_validates_missing_and_non_finite_values():
    import pandas as pd

    assert bt.get_barrier_pcts(None) == (None, None)
    assert bt.get_barrier_pcts(pd.DataFrame()) == (None, None)
    assert bt.get_barrier_pcts(pd.DataFrame({"barrier_stop_pct": [0.01]})) == (None, None)
    assert bt.get_barrier_pcts(
        pd.DataFrame({"barrier_stop_pct": [np.nan], "barrier_take_pct": [0.02]})
    ) == (None, None)
    assert bt.get_barrier_pcts(
        pd.DataFrame({"barrier_stop_pct": [0.01], "barrier_take_pct": [0.02]})
    ) == (0.01, 0.02)
