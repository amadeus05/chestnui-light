"""
Ранжирование кандидатов и ужимание маржи как в bt.py (1122–1146) и paper.py (421–441).
"""
from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T", bound=dict[str, Any])


def sort_entry_candidates(candidates: list[T]) -> list[T]:
    return sorted(
        candidates,
        key=lambda c: (c["score"], c["direction_prob"]),
        reverse=True,
    )


def allocate_entry_within_available(
    candidate_position_notional: float,
    candidate_required_margin: float,
    available_balance: float,
    leverage: float,
    *,
    min_notional: float = 10.0,
) -> tuple[float, float] | None:
    required_margin = min(float(candidate_required_margin), float(available_balance))
    position_notional = min(float(candidate_position_notional), required_margin * float(leverage))
    if position_notional < min_notional or required_margin <= 0:
        return None
    return position_notional, required_margin
