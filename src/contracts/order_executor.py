"""Абстракция исполнения торговых ордеров."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.domain.portfolio.portfolio_manager import PortfolioManager


@dataclass(frozen=True)
class OrderResult:
    """Результат исполнения ордера."""

    order_id: str
    filled_qty: float
    avg_price: float
    status: Literal["FILLED", "PARTIAL", "REJECTED"]
    commission: float


class OrderExecutor(ABC):
    """Исполнитель торговых ордеров.

    Реализации:
    - SimulationExecutor: фейковое исполнение для бэктеста/paper
    - LiveExecutor: реальные ордера на бирже
    """

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        qty: float,
        leverage: float = 1.0,
    ) -> OrderResult:
        """
        Отправка рыночного ордера.

        Args:
            symbol: Торговая пара
            side: BUY или SELL
            qty: Размер позиции в quote currency
            leverage: Плечо (для расчета маржи)

        Returns:
            OrderResult с результатом исполнения
        """
        raise NotImplementedError

    @abstractmethod
    def get_position(self, symbol: str) -> dict | None:
        """
        Получить текущую позицию на бирже (для live).

        Returns:
            dict с полями: size, entry_price, side и т.д.
            None если позиции нет
        """
        raise NotImplementedError

    @abstractmethod
    def sync_with_portfolio(self, portfolio: PortfolioManager) -> None:
        """
        Синхронизация состояния с портфолио.

        Для симулятора: ничего не делает.
        Для live: проверяет реальные позиции на бирже и обновляет портфолио.
        """
        raise NotImplementedError
