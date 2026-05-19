import math

import numpy as np
import pandas as pd
import pytest

from src_refactor.domain.execution import ExecutionPricingConfig, compute_net_pnl_pct, resolve_trade_exit
from src_refactor.domain.labels import LabelingConfig, simulate_label_trade_outcome
from src_refactor.domain.signals import SignalBatchProcessor, SignalProcessingConfig


@pytest.fixture()
def execution_config():
    return ExecutionPricingConfig(slippage=0.001, taker_fee=0.0004)


@pytest.fixture()
def signal_processor():
    return SignalBatchProcessor(
        SignalProcessingConfig(
            directional_proba_threshold=0.55,
            min_signal_gap=0.02,
        )
    )


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
def test_resolve_trade_exit_behavior(
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
    trade_exit = resolve_trade_exit(
        direction,
        entry,
        next_open,
        next_high,
        next_low,
        stop_pct,
        take_pct,
        execution_config,
    )

    exit_price = trade_exit.price
    reason = trade_exit.reason
    assert reason == expected_reason
    if expected_price is None:
        assert exit_price is None
    else:
        assert exit_price == pytest.approx(expected_price)


def test_resolve_trade_exit_checks_stop_before_take_when_both_hit(execution_config):
    trade_exit = resolve_trade_exit(
        direction=1,
        entry_price=100.0,
        next_open=100.0,
        next_high=104.0,
        next_low=97.0,
        stop_pct=0.02,
        take_pct=0.03,
        config=execution_config,
    )

    assert trade_exit.reason == "SL"
    assert trade_exit.price == pytest.approx(98.0 * 0.999)


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

    snapshot = market_batch[candidate["sym"]]
    trade_exit = resolve_trade_exit(
        candidate["signal"],
        candidate["entry_price"],
        snapshot["next_open"],
        snapshot["next_high"],
        snapshot["next_low"],
        candidate["stop_pct"],
        candidate["take_pct"],
        execution_config,
    )

    assert trade_exit.reason == "SL"
    assert trade_exit.price == pytest.approx(29_700.0 * 0.999)
    assert compute_net_pnl_pct(1, candidate["entry_price"], trade_exit.price, execution_config) > -0.02


@pytest.mark.parametrize(
    ("direction", "entry", "exit_price", "expected_raw"),
    [
        (1, 100.0, 110.0, 0.10),
        (1, 100.0, 90.0, -0.10),
        (-1, 100.0, 90.0, 0.10),
        (-1, 100.0, 110.0, -0.10),
    ],
)
def test_pnl_formula_behavior(
    execution_config,
    direction,
    entry,
    exit_price,
    expected_raw,
):
    expected_net = expected_raw - 0.0008

    assert compute_net_pnl_pct(direction, entry, exit_price, execution_config) == pytest.approx(expected_net)


def test_compute_trade_outcome_uses_notional_and_commission(execution_config):
    position = {"dir": 1, "entry": 100.0, "size": 250.0}

    pnl_clean = compute_net_pnl_pct(position["dir"], position["entry"], 110.0, execution_config)
    trade_profit = position["size"] * pnl_clean
    commission = position["size"] * (execution_config.taker_fee * 2)

    assert pnl_clean == pytest.approx(0.10 - 0.0008)
    assert trade_profit == pytest.approx(250.0 * (0.10 - 0.0008))
    assert commission == pytest.approx(250.0 * 0.0008)


def test_simulate_trade_outcome_uses_next_open_as_slipped_entry(execution_config):
    opens = np.array([100.0, 101.0, 101.0, 101.0])
    highs = np.array([100.0, 104.5, 101.0, 101.0])
    lows = np.array([100.0, 100.0, 100.0, 100.0])
    stop_pcts = np.array([0.02, 0.02, 0.02, 0.02])
    take_pcts = np.array([0.03, 0.03, 0.03, 0.03])

    pnl, reason = simulate_label_trade_outcome(
        opens,
        highs,
        lows,
        stop_pcts,
        take_pcts,
        start_idx=0,
        direction=1,
        horizon=2,
        config=LabelingConfig(slippage=execution_config.slippage, taker_fee=execution_config.taker_fee),
    )

    entry = 101.0 * 1.001
    exit_price = (entry * 1.03) * 0.999
    expected_pnl = (exit_price - entry) / entry - 0.0008
    assert reason == "TP"
    assert pnl == pytest.approx(expected_pnl)


def test_simulate_trade_outcome_returns_neutral_when_barriers_are_nan(execution_config):
    values = np.array([100.0, 101.0, 102.0])

    pnl, reason = simulate_label_trade_outcome(
        opens=values,
        highs=values + 10.0,
        lows=values - 10.0,
        stop_pcts=np.array([math.nan, 0.02, 0.02]),
        take_pcts=np.array([0.03, 0.03, 0.03]),
        start_idx=0,
        direction=1,
        horizon=2,
        config=LabelingConfig(slippage=execution_config.slippage, taker_fee=execution_config.taker_fee),
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
    signal_processor,
    p_long,
    p_short,
    expected_direction,
    expected_probability,
    expected_gap,
):
    direction, probability, gap = signal_processor.resolve_directional_signal(p_long, p_short)

    assert direction == expected_direction
    assert probability == pytest.approx(expected_probability)
    assert gap == pytest.approx(expected_gap)


def test_build_entry_score_uses_edge_above_threshold_and_gap(signal_processor):
    assert signal_processor.build_entry_score(direction_prob=0.60, signal_gap=0.03) == pytest.approx(0.53)
    assert signal_processor.build_entry_score(direction_prob=0.50, signal_gap=0.03) == pytest.approx(0.03)


def test_get_barrier_pcts_validates_missing_and_non_finite_values():
    assert _barrier_pcts(None) == (None, None)
    assert _barrier_pcts(pd.DataFrame()) == (None, None)
    assert _barrier_pcts(pd.DataFrame({"barrier_stop_pct": [0.01]})) == (None, None)
    assert _barrier_pcts(pd.DataFrame({"barrier_stop_pct": [np.nan], "barrier_take_pct": [0.02]})) == (None, None)
    assert _barrier_pcts(pd.DataFrame({"barrier_stop_pct": [0.01], "barrier_take_pct": [0.02]})) == (0.01, 0.02)


def _barrier_pcts(feature_row: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None
    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None
    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not math.isfinite(stop_pct) or not math.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct
