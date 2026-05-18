from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src_refactor.core.types import AccountSnapshot, Fill, MarketExecutionSnapshot, OrderRequest, PositionSnapshot


@dataclass(frozen=True, slots=True)
class BrokerOrderResult:
    accepted: bool = False
    cancelled: bool = False
    order: OrderRequest | None = None
    fills: tuple[Fill, ...] = ()
    reason: str | None = None

class BrokerGateway(ABC):
    """Сюда приходят ордера от стратегии и они обрабатываются в фоновом потоке
       и результаты возвращаются обратно в стратегию"""
    @abstractmethod
    def place_order(self, order: OrderRequest) -> BrokerOrderResult:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def process_market_snapshot(self, snapshot: MarketExecutionSnapshot) -> list[Fill]:
        raise NotImplementedError

    @abstractmethod
    def resolve_position_exit(self, position: PositionSnapshot, snapshot: MarketExecutionSnapshot) -> Fill | None:
        raise NotImplementedError