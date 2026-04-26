"""Исполнитель ордеров для симуляции (бэктест/paper)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.contracts import Bar, Clock, OrderExecutor, OrderResult

if TYPE_CHECKING:
    from src.domain.portfolio.portfolio_manager import PortfolioManager


def _clip(value: float, min_val: float, max_val: float) -> float:
    """Ограничивает значение диапазоном."""
    return max(min_val, min(max_val, value))


class SimulationExecutor(OrderExecutor):
    """
    Фейковое исполнение ордеров для бэктеста и paper trading.

    Не отправляет ордера на биржу. Расчет цены исполнения:
    - Entry: open текущего бара с проскальзыванием
    - Exit: цена TP/SL с проскальзыванием

    Полностью синхронен, не делает HTTP запросов.
    """

    def __init__(
        self,
        clock: Clock,
        slippage: float = 0.0003,
        taker_fee: float = 0.0004,
    ) -> None:
        self._clock = clock
        self._slippage = slippage
        self._taker_fee = taker_fee
        self._current_bar_batch: dict[str, Bar] = {}
        self._order_counter: int = 0

    def set_current_bar(self, bar_batch: dict[str, Bar]) -> None:
        """
        Устанавливает текущий бар для расчета цен исполнения.

        Вызывается TradingEngine перед обработкой каждого бара.
        """
        self._current_bar_batch = bar_batch

    def submit_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" или "SELL"
        qty: float,
        leverage: float = 1.0,
    ) -> OrderResult:
        """
        Симулирует исполнение рыночного ордера.

        Цена исполнения:
        - BUY: close * (1 + slippage) — покупаем дороже
        - SELL: close * (1 - slippage) — продаем дешевле

        Args:
            symbol: Торговая пара
            side: "BUY" (long entry / short exit) или "SELL" (short entry / long exit)
            qty: Размер позиции
            leverage: Не используется в симуляции

        Returns:
            OrderResult с рассчитанной ценой и комиссией
        """
        bar = self._current_bar_batch.get(symbol)
        if bar is None:
            raise RuntimeError(
                f"SimulationExecutor: no bar data for {symbol}. "
                "Call set_current_bar() before submit_market_order()"
            )

        # Расчет цены исполнения с проскальзыванием
        if side == "BUY":
            fill_price = bar.close * (1 + self._slippage)
        else:  # SELL
            fill_price = bar.close * (1 - self._slippage)

        # Комиссия taker
        commission = qty * self._taker_fee

        self._order_counter += 1
        order_id = f"sim_{self._clock.now().timestamp()}_{self._order_counter}"

        return OrderResult(
            order_id=order_id,
            filled_qty=qty,
            avg_price=fill_price,
            status="FILLED",
            commission=commission,
        )

    def get_position(self, symbol: str) -> dict | None:
        """
        В симуляции позиции хранятся в PortfolioManager.

        Returns None всегда — синхронизация не требуется.
        """
        return None

    def sync_with_portfolio(self, portfolio: PortfolioManager) -> None:
        """
        Ничего не делает — в симуляции нет расхождений.

        PortfolioManager является единственным источником правды.
        """
        pass

    def calculate_exit_price(
        self,
        symbol: str,
        target_price: float,
        side: str,  # "BUY" (short exit) или "SELL" (long exit)
    ) -> float:
        """
        Расчет цены выхода с учетом проскальзывания.

        Используется BarFillStrategy для TP/SL.

        Args:
            symbol: Торговая пара
            target_price: Целевая цена (TP или SL)
            side: BUY для выхода из short, SELL для выхода из long

        Returns:
            Цена исполнения с проскальзыванием
        """
        if side == "BUY":
            # Выход из short — покупаем дороже
            return target_price * (1 + self._slippage)
        else:
            # Выход из long — продаем дешевле
            return target_price * (1 - self._slippage)

    def check_bar_touch(
        self,
        bar: Bar,
        target_price: float,
        direction: str,  # "up" или "down"
    ) -> tuple[bool, float]:
        """
        Проверяет касание цены на баре.

        Args:
            bar: OHLCV бар
            target_price: Целевая цена (TP или SL)
            direction: "up" для TP long / SL short, "down" для TP short / SL long

        Returns:
            (was_touched, actual_price)
            actual_price — цена исполнения с проскальзыванием
        """
        if direction == "up":
            # Для long TP: high должен достичь target_price
            # Для short SL: high должен достичь target_price
            if bar.high >= target_price:
                # Исполняем по target_price (идеальное исполнение)
                # или open если гэп
                actual = max(target_price, bar.open)
                return True, actual
        else:  # down
            # Для long SL: low должен достичь target_price
            # Для short TP: low должен достичь target_price
            if bar.low <= target_price:
                actual = min(target_price, bar.open)
                return True, actual

        return False, 0.0
