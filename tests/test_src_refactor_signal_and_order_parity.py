import pandas as pd
import pytest

import bt

from src_refactor.core.types import MarketExecutionSnapshot, Prediction
from src_refactor.domain.signals import SignalBatchProcessor, SignalProcessingConfig
from src_refactor.domain.trading import OrderFactory


@pytest.fixture()
def signal_config(monkeypatch):
    monkeypatch.setattr(bt, "DIRECTIONAL_PROBA_THRESHOLD", 0.55)
    monkeypatch.setattr(bt, "MIN_SIGNAL_GAP", 0.02)
    return SignalProcessingConfig(
        directional_proba_threshold=0.55,
        min_signal_gap=0.02,
    )


@pytest.mark.parametrize(
    ("p_long", "p_short"),
    [
        (0.58, 0.54),
        (0.52, 0.57),
        (0.56, 0.55),
        (0.54, 0.20),
    ],
)
def test_signal_processor_directional_signal_matches_legacy_bt(signal_config, p_long, p_short):
    processor = SignalBatchProcessor(signal_config)

    assert processor.resolve_directional_signal(p_long, p_short) == pytest.approx(
        bt.resolve_directional_signal(p_long, p_short)
    )


def test_signal_processor_entry_score_matches_legacy_bt(signal_config):
    processor = SignalBatchProcessor(signal_config)

    assert processor.build_entry_score(direction_prob=0.60, signal_gap=0.03) == pytest.approx(
        bt.build_entry_score(direction_prob=0.60, signal_gap=0.03)
    )
    assert processor.build_entry_score(direction_prob=0.50, signal_gap=0.03) == pytest.approx(
        bt.build_entry_score(direction_prob=0.50, signal_gap=0.03)
    )


def test_signal_processor_candidate_sort_matches_legacy_key(signal_config):
    processor = SignalBatchProcessor(signal_config)
    predictions = [
        _prediction("BTC/USDT", p_long=0.58, p_short=0.40),
        _prediction("ETH/USDT", p_long=0.59, p_short=0.55),
        _prediction("SOL/USDT", p_long=0.57, p_short=0.10),
    ]

    candidates = processor.build_candidates(predictions)
    expected_symbols = [
        item.symbol
        for item in sorted(candidates, key=lambda item: (item.score, item.direction_prob), reverse=True)
    ]

    assert [candidate.symbol for candidate in candidates] == expected_symbols


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"barrier_stop_pct": 0.02},
        {"barrier_stop_pct": float("nan"), "barrier_take_pct": 0.04},
        {"barrier_stop_pct": 0.02, "barrier_take_pct": float("inf")},
    ],
)
def test_signal_processor_rejects_candidates_without_valid_barriers(signal_config, raw):
    prediction = _prediction("BTC/USDT", p_long=0.90, p_short=0.10, raw=raw)

    assert SignalBatchProcessor(signal_config).build_candidates([prediction]) == []


def test_order_factory_builds_legacy_market_order_shape(signal_config):
    candidate = SignalBatchProcessor(signal_config).build_candidates(
        [_prediction("BTC/USDT", p_long=0.90, p_short=0.10)]
    )[0]
    snapshot = MarketExecutionSnapshot(
        symbol="BTC/USDT",
        current_timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        next_timestamp=pd.Timestamp("2025-01-01 01:00:00"),
        current_close=100.0,
        next_open=100.0,
        next_high=101.0,
        next_low=99.0,
    )

    order = OrderFactory().from_signal_candidate(
        candidate,
        position_notional=250.0,
        snapshot=snapshot,
    )

    assert order.order_id == f"BTC/USDT:{snapshot.next_timestamp.value}:buy"
    assert order.symbol == "BTC/USDT"
    assert order.side == "buy"
    assert order.order_type == "market"
    assert order.quantity == pytest.approx(2.5)
    assert order.stop_pct == 0.02
    assert order.take_pct == 0.04
    assert order.created_at == snapshot.current_timestamp


def _prediction(
    symbol: str,
    *,
    p_long: float,
    p_short: float,
    raw: dict[str, float] | None = None,
) -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        symbol=symbol,
        timeframe="1h",
        model_id="model",
        direction=0,
        confidence=max(p_long, p_short),
        proba_long=p_long,
        proba_short=p_short,
        raw=raw if raw is not None else {"barrier_stop_pct": 0.02, "barrier_take_pct": 0.04},
    )
