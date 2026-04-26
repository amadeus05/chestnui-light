"""Заготовка под правила из structure (2).md; текущий легаси-сайзинг — в RiskManager."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class RiskRule(ABC):
    @abstractmethod
    def evaluate(self, context: Any) -> Any:
        raise NotImplementedError
