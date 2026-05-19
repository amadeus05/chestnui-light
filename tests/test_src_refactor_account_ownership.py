import pandas as pd
import pytest

from src_refactor.core.contracts.broker_gateway import BrokerGateway, BrokerOrderResult
from src_refactor.core.types import (
    AccountSnapshot,
    Fill,
    MarketExecutionSnapshot,
    OrderRequest,
    OrderSnapshot,
    PositionSnapshot,
    Prediction,
)
from src_refactor.domain.portfolio.portfolio_manager import PortfolioManager
from src_refactor.domain.risk.risk_manager import RiskManager
from src_refactor.domain.signals import SignalBatchProcessor
from src_refactor.domain.trading import TradingEngine
from src_refactor.infrastructure.exchanges.simulation import ExchangeSimulator


def _snapshot(symbol: str = "BTC/USDT") -> MarketExecutionSnapshot:
    return MarketExecutionSnapshot(
        symbol=symbol,
        current_timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        next_timestamp=pd.Timestamp("2025-01-01 01:00:00"),
        current_close=100.0,
        next_open=100.0,
        next_high=101.0,
        next_low=99.0,
    )


def _candidate(symbol: str = "BTC/USDT"):
    prediction = Prediction(
        timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        symbol=symbol,
        timeframe="1h",
        model_id="model",
        direction=0,
        confidence=0.9,
        proba_long=0.9,
        proba_short=0.1,
        stop_pct=0.02,
        take_pct=0.04,
    )
    return SignalBatchProcessor().build_candidates([prediction])[0]


def _order(symbol: str = "BTC/USDT") -> OrderRequest:
    return OrderRequest(
        order_id=f"{symbol}:order",
        symbol=symbol,
        side="buy",
        order_type="market",
        quantity=1.0,
        created_at=pd.Timestamp("2025-01-01 00:00:00"),
    )


class StaticAccountBroker(BrokerGateway):
    def __init__(self, account: AccountSnapshot) -> None:
        self.account = account
        self.orders: list[OrderRequest] = []

    def place_order(self, order: OrderRequest) -> BrokerOrderResult:
        self.orders.append(order)
        return BrokerOrderResult(accepted=True, order=OrderSnapshot.from_request(order))

    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        return BrokerOrderResult(cancelled=False)

    def get_account_snapshot(self) -> AccountSnapshot:
        return self.account

    def process_market_snapshot(self, snapshot: MarketExecutionSnapshot) -> list[Fill]:
        return []

    def resolve_position_exit(self, position: PositionSnapshot, snapshot: MarketExecutionSnapshot) -> Fill | None:
        return None


def test_exchange_simulator_account_snapshot_tracks_portfolio_provider():
    broker = ExchangeSimulator()
    portfolio = PortfolioManager(initial_balance=250.0)
    broker.set_account_snapshot_provider(lambda: portfolio.account_snapshot({"BTC/USDT": 101.0}))
    portfolio.open_position(
        symbol="BTC/USDT",
        direction=1,
        entry_price=100.0,
        position_notional=50.0,
        required_margin=50.0,
        stop_pct=0.02,
        take_pct=0.04,
        opened_at=pd.Timestamp("2025-01-01 01:00:00"),
    )

    account = broker.get_account_snapshot()

    assert account.balance == 250.0
    assert account.used_margin == 50.0
    assert account.equity > 250.0
    assert set(account.positions) == {"BTC/USDT"}


def test_portfolio_exposes_position_state_without_leaking_storage():
    portfolio = PortfolioManager(initial_balance=250.0)
    portfolio.open_position(
        symbol="BTC/USDT",
        direction=1,
        entry_price=100.0,
        position_notional=50.0,
        required_margin=25.0,
        stop_pct=0.02,
        take_pct=0.04,
        opened_at=pd.Timestamp("2025-01-01 01:00:00"),
    )

    snapshots = portfolio.position_snapshots()

    assert portfolio.has_position("BTC/USDT")
    assert not portfolio.has_position("ETH/USDT")
    assert set(snapshots) == {"BTC/USDT"}
    assert snapshots["BTC/USDT"].entry_price == 100.0


def test_portfolio_opens_position_from_entry_fill():
    portfolio = PortfolioManager(initial_balance=250.0)
    fill = Fill(
        fill_id="BTC/USDT:entry",
        order_id="BTC/USDT:order",
        symbol="BTC/USDT",
        side="buy",
        price=101.0,
        quantity=0.5,
        fee=0.0,
        reason="ENTRY",
        timestamp=pd.Timestamp("2025-01-01 01:00:00"),
    )

    position = portfolio.open_position_from_fill(
        fill,
        direction=1,
        position_notional=50.0,
        required_margin=25.0,
        stop_pct=0.02,
        take_pct=0.04,
    )

    assert position.symbol == "BTC/USDT"
    assert position.entry_price == 101.0
    assert position.opened_at == fill.timestamp
    assert portfolio.position_snapshot("BTC/USDT") is not None


def test_portfolio_rejects_non_entry_fill_for_opening_position():
    portfolio = PortfolioManager(initial_balance=250.0)
    fill = Fill(
        fill_id="BTC/USDT:exit",
        order_id="BTC/USDT:order",
        symbol="BTC/USDT",
        side="sell",
        price=99.0,
        quantity=0.5,
        fee=0.0,
        reason="SL",
        timestamp=pd.Timestamp("2025-01-01 01:00:00"),
    )

    with pytest.raises(ValueError):
        portfolio.open_position_from_fill(
            fill,
            direction=1,
            position_notional=50.0,
            required_margin=25.0,
            stop_pct=0.02,
            take_pct=0.04,
        )


def test_portfolio_closes_position_from_fill():
    portfolio = PortfolioManager(initial_balance=250.0)
    portfolio.open_position(
        symbol="BTC/USDT",
        direction=1,
        entry_price=100.0,
        position_notional=50.0,
        required_margin=25.0,
        stop_pct=0.02,
        take_pct=0.04,
        opened_at=pd.Timestamp("2025-01-01 01:00:00"),
    )
    fill = Fill(
        fill_id="BTC/USDT:exit",
        order_id="BTC/USDT:order",
        symbol="BTC/USDT",
        side="sell",
        price=104.0,
        quantity=0.5,
        fee=0.0,
        reason="TP",
        timestamp=pd.Timestamp("2025-01-01 02:00:00"),
    )

    trade = portfolio.close_position_from_fill(fill)

    assert trade is not None
    assert trade.symbol == "BTC/USDT"
    assert trade.reason == "TP"
    assert trade.exit_price == 104.0
    assert portfolio.position_snapshot("BTC/USDT") is None


def test_exchange_simulator_exposes_order_snapshots_not_strategy_requests():
    broker = ExchangeSimulator()

    result = broker.place_order(_order())
    account = broker.get_account_snapshot()

    assert result.accepted
    assert isinstance(result.order, OrderSnapshot)
    assert isinstance(account.open_orders["BTC/USDT:order"], OrderSnapshot)
    assert account.open_orders["BTC/USDT:order"].status == "open"


def test_exchange_simulator_returns_terminal_order_snapshots():
    broker = ExchangeSimulator()
    broker.place_order(_order())

    cancelled = broker.cancel_order("BTC/USDT:order")
    rejected = broker.place_order(
        OrderRequest(
            order_id="BTC/USDT:unsupported",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",  # type: ignore[arg-type]
            quantity=1.0,
        )
    )

    assert cancelled.cancelled
    assert cancelled.order is not None
    assert cancelled.order.status == "cancelled"
    assert rejected.order is not None
    assert rejected.order.status == "rejected"


def test_exchange_simulator_resolves_final_mark_close_with_legacy_slippage():
    broker = ExchangeSimulator()
    position = PositionSnapshot(
        symbol="BTC/USDT",
        direction=1,
        quantity=2.0,
        entry_price=100.0,
        opened_at=pd.Timestamp("2025-01-01 00:00:00"),
    )

    fill = broker.resolve_position_mark_close(
        position,
        timestamp=pd.Timestamp("2025-01-02 00:00:00"),
        mark_price=110.0,
        reason="FINAL",
    )

    assert fill.reason == "FINAL"
    assert fill.side == "sell"
    assert fill.price == 110.0 * (1 - broker.pricing.slippage)


def test_trading_engine_uses_broker_account_for_sizing():
    broker = ExchangeSimulator()
    portfolio = PortfolioManager(initial_balance=100.0)
    engine = TradingEngine(
        broker=broker,
        portfolio=portfolio,
        risk=RiskManager(),
    )

    result = engine.on_market_batch(
        bar_index=0,
        snapshots={"BTC/USDT": _snapshot()},
        candidates=[_candidate()],
    )

    assert result.account is not None
    assert result.account.balance == 100.0
    assert len(result.opened_orders) == 1
    assert result.opened_orders[0].quantity > 0


def test_trading_engine_respects_existing_broker_position_when_portfolio_is_empty():
    account = AccountSnapshot(
        balance=1_000.0,
        equity=1_000.0,
        used_margin=100.0,
        positions={
            "BTC/USDT": PositionSnapshot(
                symbol="BTC/USDT",
                direction=1,
                quantity=1.0,
                entry_price=100.0,
                stop_pct=0.02,
                take_pct=0.04,
                opened_at=pd.Timestamp("2025-01-01 00:00:00"),
            )
        },
    )
    broker = StaticAccountBroker(account)
    engine = TradingEngine(
        broker=broker,
        portfolio=PortfolioManager(initial_balance=100.0),
        risk=RiskManager(),
    )

    result = engine.on_market_batch(
        bar_index=0,
        snapshots={"BTC/USDT": _snapshot()},
        candidates=[_candidate()],
    )

    assert result.account == account
    assert result.opened_orders == ()
    assert broker.orders == []
