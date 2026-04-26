"""
Ограничения на число новых/открытых позиций за бар — как в bt.py (1130–1136).
"""
from __future__ import annotations

from .models.constraints import BarEntryConstraints


class ExecutionService:
    def __init__(self, constraints: BarEntryConstraints) -> None:
        self._c = constraints

    @property
    def constraints(self) -> BarEntryConstraints:
        return self._c

    def may_open_more(self, opened_this_bar: int, open_positions_count: int) -> bool:
        if opened_this_bar >= self._c.max_new_positions_per_bar:
            return False
        if open_positions_count >= self._c.max_open_positions:
            return False
        return True
