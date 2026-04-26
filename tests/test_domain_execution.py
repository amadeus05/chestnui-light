"""Тесты order_router и ExecutionService."""
from __future__ import annotations

import pytest

from src.domain.execution import (
    BarEntryConstraints,
    ExecutionService,
    allocate_entry_within_available,
    sort_entry_candidates,
)


def test_sort_entry_candidates_primary_score_then_prob():
    cands = [
        {"score": 1.0, "direction_prob": 0.9, "id": "a"},
        {"score": 2.0, "direction_prob": 0.1, "id": "b"},
        {"score": 2.0, "direction_prob": 0.8, "id": "c"},
    ]
    out = sort_entry_candidates(cands)
    assert [x["id"] for x in out] == ["c", "b", "a"]


def test_allocate_entry_within_available_bt_shape():
    out = allocate_entry_within_available(
        candidate_position_notional=500.0,
        candidate_required_margin=200.0,
        available_balance=50.0,
        leverage=10.0,
        min_notional=10.0,
    )
    assert out is not None
    n, m = out
    assert m == pytest.approx(50.0)
    assert n == pytest.approx(500.0)


def test_allocate_entry_rejects_below_min_notional():
    assert (
        allocate_entry_within_available(9.0, 5.0, 100.0, 3.0, min_notional=10.0) is None
    )
    assert allocate_entry_within_available(10.0, 0.0, 100.0, 3.0, min_notional=10.0) is None


def test_execution_service_may_open_more():
    svc = ExecutionService(BarEntryConstraints(max_new_positions_per_bar=2, max_open_positions=5))
    assert svc.may_open_more(0, 4)
    assert svc.may_open_more(1, 4)
    assert not svc.may_open_more(2, 4)
    assert not svc.may_open_more(0, 5)
