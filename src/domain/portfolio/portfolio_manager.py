"""
Состояние портфеля как в bt.py: balance, used_margin, positions[sym] -> dict | None.
Эквити и исход сделок — только через pnl.py (идентичные формулы).
"""
from __future__ import annotations

from typing import Any

from .models.account_state import AccountState
from .models.balance import MarginCashBalance
from .models.pnl import compute_portfolio_equity, compute_trade_outcome
from .models.portfolio_state import PortfolioState


class PortfolioManager:
    def __init__(
        self,
        initial_balance: float,
        *,
        taker_com: float = 0.0004,
    ) -> None:
        self._taker_com = float(taker_com)
        cash = MarginCashBalance(float(initial_balance))
        acct = AccountState(peak_equity=float(initial_balance))
        self._state = PortfolioState(balance=cash, account=acct)

    @property
    def taker_com(self) -> float:
        return self._taker_com

    @property
    def state(self) -> PortfolioState:
        return self._state

    @property
    def cash(self) -> float:
        return self._state.balance.cash

    @property
    def used_margin(self) -> float:
        return self._state.balance.used_margin

    def available_balance(self) -> float:
        return self._state.balance.available

    def position(self, symbol: str) -> dict[str, Any] | None:
        return self._state.positions.get(symbol)

    def set_position(self, symbol: str, pos: dict[str, Any] | None) -> None:
        self._state.set_slot(symbol, pos)

    def ensure_symbol_slot(self, symbol: str) -> None:
        if symbol not in self._state.positions:
            self._state.positions[symbol] = None

    def open_position(
        self,
        symbol: str,
        *,
        trade_number: int,
        direction: int,
        entry_price: float,
        position_notional: float,
        required_margin: float,
        stop_pct: float,
        take_pct: float,
        ts_open: Any,
    ) -> None:
        self._state.balance.add_used_margin(required_margin)
        self.set_position(
            symbol,
            {
                "trade_number": trade_number,
                "dir": direction,
                "entry": float(entry_price),
                "size": float(position_notional),
                "margin": float(required_margin),
                "stop_pct": float(stop_pct),
                "take_pct": float(take_pct),
                "ts_open": ts_open,
            },
        )

    def close_position_at_price(self, symbol: str, exit_price: float) -> tuple[float, float, float]:
        pos = self.position(symbol)
        if pos is None:
            raise ValueError(f"no open position for {symbol}")
        pnl_clean, trade_profit, commission = compute_trade_outcome(
            pos, float(exit_price), taker_com=self._taker_com
        )
        self._state.balance.release_margin(float(pos["margin"]))
        self._state.balance.add_trade_pnl(trade_profit)
        self.set_position(symbol, None)
        return pnl_clean, trade_profit, commission

    def equity(self, mark_prices: dict[str, float]) -> float:
        return compute_portfolio_equity(
            self._state.balance.cash,
            self._state.positions,
            mark_prices,
            taker_com=self._taker_com,
        )

    def observe_equity(self, mark_prices: dict[str, float]) -> float:
        eq = self.equity(mark_prices)
        self._state.account.observe_equity(eq)
        return eq

    @property
    def peak_equity(self) -> float:
        return self._state.account.peak_equity

    @property
    def max_drawdown_pct(self) -> float:
        return self._state.account.max_drawdown_pct
