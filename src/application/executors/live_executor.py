"""Исполнитель ордеров для live trading — реальная биржа."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from src.contracts import Clock, OrderExecutor, OrderResult

if TYPE_CHECKING:
    from src.domain.portfolio.portfolio_manager import PortfolioManager


class LiveExecutor(OrderExecutor):
    """
    Реальное исполнение ордеров на бирже.

    Требует exchange с методами:
    - place_market_order(symbol, side, qty) -> dict
    - get_position(symbol) -> dict | None
    - get_last_price(symbol) -> float | None  (опционально, для reconcile)

    Синхронизирует состояние с биржей при каждом тике.
    """

    def __init__(
        self,
        exchange: Any,
        clock: Clock,
    ) -> None:
        self._exchange = exchange
        self._clock = clock

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        leverage: float = 1.0,
    ) -> OrderResult:
        """
        Отправляет реальный рыночный ордер на биржу.

        Args:
            symbol: Торговая пара (например "BTCUSDT")
            side: "BUY" для long entry / short exit, "SELL" для short entry / long exit
            qty: Размер позиции в quote currency
            leverage: Плечо

        Returns:
            OrderResult с реальными данными исполнения
        """
        if not hasattr(self._exchange, "place_market_order"):
            raise RuntimeError(
                f"Exchange {type(self._exchange).__name__} does not implement "
                "place_market_order(). Подключи торговый клиент биржи."
            )

        normalized_symbol = (
            self._exchange.normalize_symbol(symbol)
            if hasattr(self._exchange, "normalize_symbol")
            else symbol
        )

        response = self._exchange.place_market_order(
            symbol=normalized_symbol,
            side=side,
            qty=qty,
        )

        order_id = response.get("order_id", response.get("id", "unknown"))
        filled_qty = float(response.get("filled_qty", response.get("executedQty", qty)))
        avg_price = float(response.get("avg_price", response.get("avgPrice", 0.0)))
        commission = float(response.get("commission", 0.0))

        status_str = response.get("status", "FILLED")
        if status_str in {"FILLED", "CLOSED"}:
            status = "FILLED"
        elif status_str in {"PARTIALLY_FILLED", "PARTIAL"}:
            status = "PARTIAL"
        else:
            status = "REJECTED"

        print(
            f"[LiveExecutor] {side} {qty:.2f} {symbol} @ {avg_price:.4f} "
            f"(status: {status})"
        )

        return OrderResult(
            order_id=order_id,
            filled_qty=filled_qty,
            avg_price=avg_price,
            status=status,  # type: ignore[arg-type]
            commission=commission,
        )

    def get_position(self, symbol: str) -> dict | None:
        """
        Получает реальную позицию с биржи.

        Returns:
            dict с полями: size, entry_price, side, unrealized_pnl
            None если позиции нет или exchange не поддерживает этот метод
        """
        if not hasattr(self._exchange, "get_position"):
            return None
        try:
            normalized = (
                self._exchange.normalize_symbol(symbol)
                if hasattr(self._exchange, "normalize_symbol")
                else symbol
            )
            return self._exchange.get_position(normalized)
        except Exception as e:
            print(f"[LiveExecutor] get_position({symbol}) error: {e}")
            return None

    def sync_with_portfolio(self, portfolio: "PortfolioManager") -> None:
        """
        Синхронизирует локальное портфолио с реальным состоянием биржи.

        Сценарии:
        1. Позиция есть на бирже, но нет локально — ручной вход, предупреждение.
        2. Позиция есть локально, но нет на бирже — закрыта снаружи (ликвидация / ручное закрытие).
        3. Расхождение в размере — предупреждение.

        Если exchange не поддерживает get_position — sync пропускается.
        """
        if not hasattr(self._exchange, "get_position"):
            return

        for symbol in list(portfolio.state.positions.keys()):
            local_pos = portfolio.position(symbol)
            exchange_pos = self.get_position(symbol)

            if local_pos is None and exchange_pos is None:
                continue

            if local_pos is None and exchange_pos is not None:
                # Позиция открыта на бирже вручную — бот её не открывал
                print(
                    f"[LiveExecutor] ⚠️  Sync {symbol}: найдена позиция на бирже "
                    f"(size={exchange_pos.get('size', '?')}, side={exchange_pos.get('side', '?')}), "
                    "но в локальном портфолио её нет (ручной вход?)."
                )

            elif local_pos is not None and exchange_pos is None:
                # Позиция закрыта снаружи: ликвидация или ручное закрытие
                print(
                    f"[LiveExecutor] ⚠️  Sync {symbol}: позиция закрыта на бирже "
                    f"(ликвидация или ручное закрытие). Закрываю локальную позицию."
                )
                exit_price = self._get_approx_exit_price(symbol, local_pos)
                if exit_price is not None:
                    try:
                        portfolio.close_position_at_price(symbol, exit_price)
                        print(f"[LiveExecutor] Sync {symbol}: локальная позиция закрыта @ {exit_price:.4f}")
                    except Exception as e:
                        print(f"[LiveExecutor] Sync {symbol}: ошибка закрытия локальной позиции: {e}")
                else:
                    print(f"[LiveExecutor] Sync {symbol}: не удалось получить цену для закрытия локальной позиции.")

            elif local_pos is not None and exchange_pos is not None:
                # Обе стороны: проверяем расхождение в размере
                local_size = float(local_pos.get("size", 0.0))
                exchange_size = float(exchange_pos.get("size", 0.0))
                if abs(local_size - exchange_size) > 1e-4:
                    print(
                        f"[LiveExecutor] ⚠️  Sync {symbol}: расхождение размера "
                        f"local={local_size:.4f} vs exchange={exchange_size:.4f}"
                    )

    def close_position(self, symbol: str) -> OrderResult:
        """
        Закрывает позицию рыночным ордером.

        Удобный метод для экстренного закрытия.
        """
        pos = self.get_position(symbol)
        if pos is None:
            raise RuntimeError(f"No position to close for {symbol}")

        size = pos.get("size", 0.0)
        side = pos.get("side", "")

        if side == "LONG":
            close_side = "SELL"
        elif side == "SHORT":
            close_side = "BUY"
        else:
            raise RuntimeError(f"Unknown position side: {side}")

        return self.submit_market_order(symbol, close_side, size)

    # ── вспомогательные ───────────────────────────────────────────────────────

    def _get_approx_exit_price(
        self, symbol: str, local_pos: dict
    ) -> float | None:
        """
        Пытается получить текущую цену для reconcile.

        Приоритет:
        1. exchange.get_last_price(symbol)
        2. exchange.get_ticker(symbol)["last_price"]
        3. Цена входа из локальной позиции (fallback, PnL будет 0)
        """
        # Попытка 1 — прямой метод
        if hasattr(self._exchange, "get_last_price"):
            try:
                price = self._exchange.get_last_price(symbol)
                if price is not None and float(price) > 0:
                    return float(price)
            except Exception:
                pass

        # Попытка 2 — через ticker
        if hasattr(self._exchange, "get_ticker"):
            try:
                ticker = self._exchange.get_ticker(symbol)
                if ticker and "last_price" in ticker:
                    return float(ticker["last_price"])
            except Exception:
                pass

        # Попытка 3 — последний бар через fetch_klines
        if hasattr(self._exchange, "fetch_klines"):
            try:
                now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
                lookback_ms = 60_000 * 5  # 5 минут
                klines = self._exchange.fetch_klines(
                    symbol, "1m", now_ms - lookback_ms, now_ms
                )
                if klines:
                    return float(klines[-1].close)
            except Exception:
                pass

        # Fallback — цена входа (PnL будет 0, но позиция закроется корректно)
        entry = local_pos.get("entry")
        if entry is not None:
            print(
                f"[LiveExecutor] _get_approx_exit_price({symbol}): "
                "используем цену входа как fallback — PnL reconcile будет 0"
            )
            return float(entry)

        return None
