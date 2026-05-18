from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src_refactor.core.contracts.broker_gateway import BrokerGateway, BrokerOrderResult
from src_refactor.core.types import AccountSnapshot, Fill, MarketExecutionSnapshot, OrderRequest, PositionSnapshot
from src_refactor.domain.execution import ExecutionPricingConfig
from src_refactor.infrastructure.exchanges.simulation.fill_model import CandleFillModel


@dataclass(slots=True)
class ExchangeSimulator:
    """Shared simulated exchange for backtest and paper modes."""

    pricing: ExecutionPricingConfig = field(default_factory=ExecutionPricingConfig)
    open_orders: dict[str, OrderRequest] = field(init=False, default_factory=dict)
    fills: list[Fill] = field(init=False, default_factory=list)
    fill_model: CandleFillModel = field(init=False)
    account_snapshot_provider: Callable[[], AccountSnapshot] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.fill_model = CandleFillModel(self.pricing)

    def place_order(self, order: OrderRequest) -> BrokerOrderResult:
        if order.order_type != "market":
            rejected = self._replace_order_status(order, "rejected")
            return BrokerOrderResult(accepted=False, order=rejected, reason="Only market orders are supported.")
        self.open_orders[order.order_id] = order
        return BrokerOrderResult(accepted=True, order=order)

    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        order = self.open_orders.pop(order_id, None)
        if order is None:
            return BrokerOrderResult(cancelled=False, reason="Order not found.")
        return BrokerOrderResult(cancelled=True, order=self._replace_order_status(order, "cancelled"))

    def get_account_snapshot(self) -> AccountSnapshot:
        if self.account_snapshot_provider is None:
            return AccountSnapshot(balance=0.0, equity=0.0, used_margin=0.0, open_orders=dict(self.open_orders))
        snapshot = self.account_snapshot_provider()
        return AccountSnapshot(
            balance=snapshot.balance,
            equity=snapshot.equity,
            used_margin=snapshot.used_margin,
            positions=dict(snapshot.positions),
            open_orders={**snapshot.open_orders, **self.open_orders},
        )

    def set_account_snapshot_provider(self, provider: Callable[[], AccountSnapshot]) -> None:
        self.account_snapshot_provider = provider

    def process_market_snapshot(self, snapshot: MarketExecutionSnapshot) -> list[Fill]:
        fills: list[Fill] = []
        fills.extend(self._fill_open_orders(snapshot))
        self.fills.extend(fills)
        return fills

    def resolve_position_exit(self, position: PositionSnapshot, snapshot: MarketExecutionSnapshot) -> Fill | None:
        fill = self.fill_model.resolve_position_exit(
            order_id=f"position:{position.symbol}",
            position=position,
            snapshot=snapshot,
        )
        if fill is not None:
            self.fills.append(fill)
        return fill

    def _fill_open_orders(self, snapshot: MarketExecutionSnapshot) -> list[Fill]:
        fills: list[Fill] = []
        for order_id, order in list(self.open_orders.items()):
            if order.symbol != snapshot.symbol:
                continue
            fill = self.fill_model.fill_market_order(order, snapshot)
            fills.append(fill)
            del self.open_orders[order_id]
        return fills

    @staticmethod
    def _replace_order_status(order: OrderRequest, status: str) -> OrderRequest:
        return OrderRequest(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            reduce_only=order.reduce_only,
            stop_pct=order.stop_pct,
            take_pct=order.take_pct,
            created_at=order.created_at,
            status=status,  # type: ignore[arg-type]
        )
