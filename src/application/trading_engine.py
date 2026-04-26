"""TradingEngine — единый движок для paper и live; опционально replay по барам.

Оффлайн-бэктест с семантикой «close(t) → исполнение на t+1» — это **не** ветка
``ReplayDataFeed`` по умолчанию: используй ``run_backtest.py`` /
``BacktestOrchestrator`` → ``portfolio_backtest_runner``.

Тот же движок с той же семантикой t→t+1 можно собрать как
``PortfolioReplayFeed`` + ``PortfolioBacktestContext`` + ``trading_engine_portfolio_backtest``.

Ветка ``ReplayDataFeed`` + этот класс — пошаговый цикл по общим таймстампам без
обязательного сдвига decision/exec; не считай её основным бэктестом проекта.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd

from src.application.backtest.portfolio_backtest_context import PortfolioBacktestContext
from src.application.feeds.portfolio_replay_feed import PortfolioReplayFeed
from src.contracts import Bar, Clock, DataFeed, OrderExecutor

if TYPE_CHECKING:
    from src.domain.execution.execution_service import ExecutionService
    from src.domain.portfolio.portfolio_manager import PortfolioManager
    from src.domain.risk.risk_manager import RiskManager
    from src.domain.signals.signal_brain import SignalBrain


@dataclass
class TradeRecord:
    """Запись о сделке."""

    trade_number: int
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    size: float
    margin: float
    pnl_pct: float
    pnl_abs: float
    commission: float
    reason: Literal["TP", "SL", "FINAL", "MANUAL"]
    stop_pct: float
    take_pct: float


class TradingEngine:
    """
    Единый торговый движок для всех режимов.

    Использует Dependency Injection для всех зависимостей:
    - Clock: время (Replay для бэктеста, Wall для live)
    - DataFeed: источник баров (Replay или Live)
    - OrderExecutor: исполнение (Simulation или Live)
    - PortfolioManager: состояние портфолио
    - RiskManager: сайзинг и лимиты
    - ExecutionService: ограничения на входы
    - SignalBrain: ML сигналы

    Главный цикл:
    1. Получить batch баров от DataFeed
    2. Синхронизировать с биржей (для live)
    3. Phase 1: проверка TP/SL, закрытие позиций
    4. Phase 2: поиск входов, открытие новых позиций
    5. Логирование

    Портфельный бэктест со сдвигом баров:
    ``portfolio_replay_feed=PortfolioReplayFeed(...)``, ``portfolio_backtest_context=PortfolioBacktestContext(...)``,
    ``data_feed=None``. Реализация цикла: ``trading_engine_portfolio_backtest``.
    """

    def __init__(
        self,
        clock: Clock,
        data_feed: DataFeed | None,
        order_executor: OrderExecutor,
        portfolio: PortfolioManager,
        risk_manager: RiskManager,
        execution_service: ExecutionService,
        signal_brain: SignalBrain,
        symbols: list[str],
        timeframe: str,
        # Настройки поведения
        slippage: float = 0.0003,
        taker_fee: float = 0.0004,
        use_open_price_for_entry: bool = True,
        # Лимиты
        sl_cooldown_bars: int = 0,
        max_sl_per_day: int = 0,
        # Callbacks
        on_bar: callable | None = None,
        on_trade: callable | None = None,
        portfolio_replay_feed: PortfolioReplayFeed | None = None,
        portfolio_backtest_context: PortfolioBacktestContext | None = None,
    ) -> None:
        self._clock = clock
        self._feed = data_feed
        self._executor = order_executor
        self._portfolio = portfolio
        self._risk = risk_manager
        self._execution = execution_service
        self._brain = signal_brain
        self._symbols = symbols
        self._timeframe = timeframe

        # Настройки
        self._slippage = slippage
        self._taker_fee = taker_fee
        self._use_open_price = use_open_price_for_entry
        self._sl_cooldown_bars = sl_cooldown_bars
        self._max_sl_per_day = max_sl_per_day

        # Callbacks
        self._on_bar = on_bar
        self._on_trade = on_trade

        # Состояние движка
        self._trade_counter: int = 1
        self._consecutive_losses: int = 0
        self._opened_this_bar: int = 0
        self._stop_cooldown_until: dict[str, pd.Timestamp] = {}
        self._daily_sl_count: int = 0
        self._current_trade_day: pd.Timestamp | None = None
        self._is_running: bool = False
        self._current_bar_batch: dict[str, Bar] = {}

        # История
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[tuple[pd.Timestamp, float]] = []

        self._portfolio_replay_feed = portfolio_replay_feed
        self._portfolio_backtest_context = portfolio_backtest_context
        self._portfolio_universe: list[str] = []
        self._portfolio_step_i: int = -1
        self._stop_cooldown_until_step: dict[str, int] = {}

    @property
    def trades(self) -> list[TradeRecord]:
        """История всех сделок."""
        return self._trades.copy()

    @property
    def portfolio(self) -> PortfolioManager:
        return self._portfolio

    @property
    def equity_curve(self) -> list[tuple[pd.Timestamp, float]]:
        """История equity."""
        return self._equity_curve.copy()

    def run(self) -> None:
        """Главный цикл движка."""
        if self._portfolio_replay_feed is not None:
            self._run_portfolio_backtest_replay()
            return

        if self._feed is None:
            raise ValueError("data_feed обязателен, если не задан portfolio_replay_feed")

        print("=" * 60)
        print("TradingEngine started")
        print(f"  Symbols: {', '.join(self._symbols)}")
        print(f"  Timeframe: {self._timeframe}")
        print(f"  Clock: {type(self._clock).__name__}")
        print(f"  Feed: {type(self._feed).__name__}")
        print(f"  Executor: {type(self._executor).__name__}")
        print("=" * 60)

        # Инициализация
        self._feed.initialize(self._symbols, self._timeframe)
        self._is_running = True

        # Инициализация позиций
        for symbol in self._symbols:
            self._portfolio.ensure_symbol_slot(symbol)

        # Первый equity point
        initial_equity = self._portfolio.cash
        self._equity_curve.append((self._clock.now(), initial_equity))

        try:
            while self._is_running:
                # 1. Получаем следующий batch баров
                bar_batch = self._feed.next_bar_batch()
                if bar_batch is None:
                    print("[TradingEngine] No more data, stopping")
                    break

                self._current_bar_batch = bar_batch
                bar_time = list(bar_batch.values())[0].timestamp

                # Обновляем симулятор ценами текущего бара
                if hasattr(self._executor, "set_current_bar"):
                    self._executor.set_current_bar(bar_batch)

                # 2. Синхронизация с биржей (для live)
                self._executor.sync_with_portfolio(self._portfolio)

                # Обновляем дневные счетчики
                self._update_daily_counters(bar_time)

                # 3. Обновляем equity
                mark_prices = self._feed.get_mark_prices()
                equity = self._portfolio.observe_equity(mark_prices)
                self._equity_curve.append((bar_time, equity))

                # 4. Phase 1: обработка выходов (TP/SL)
                self._process_exits(bar_batch)

                # 5. Phase 2: обработка входов
                self._process_entries(bar_batch)

                # Сброс счетчиков на новом баре
                self._opened_this_bar = 0

                # Callback
                if self._on_bar:
                    self._on_bar(bar_time, bar_batch, equity)

        except KeyboardInterrupt:
            print("[TradingEngine] Interrupted by user")
        except Exception as e:
            print(f"[TradingEngine] Error: {e}")
            raise
        finally:
            self._is_running = False
            self._finalize()

    def _run_portfolio_backtest_replay(self) -> None:
        """Портфельный бэктест через PortfolioReplayFeed (close decision → exec bar)."""
        if self._portfolio_backtest_context is None:
            raise ValueError("portfolio_backtest_context обязателен при portfolio_replay_feed")

        from src.application.trading_engine_portfolio_backtest import (
            finalize_portfolio_backtest,
            run_portfolio_replay_loop,
        )

        ctx = self._portfolio_backtest_context
        uni = list(ctx.all_raw.keys())
        print("=" * 60)
        print("TradingEngine started (portfolio backtest replay)")
        print(f"  Symbols: {', '.join(uni)}")
        print(f"  Timeframe: {self._timeframe}")
        print(f"  Clock: {type(self._clock).__name__}")
        print(f"  Feed: PortfolioReplayFeed")
        print(f"  Executor: {type(self._executor).__name__}")
        print("=" * 60)

        self._is_running = True
        try:
            run_portfolio_replay_loop(self)
        except KeyboardInterrupt:
            print("[TradingEngine] Interrupted by user")
        except Exception as e:
            print(f"[TradingEngine] Error: {e}")
            raise
        finally:
            self._is_running = False
            print("\n[TradingEngine] Finalizing...")
            finalize_portfolio_backtest(self)

    def stop(self) -> None:
        """Остановка движка (для внешнего управления)."""
        self._is_running = False

    def _update_daily_counters(self, bar_time: pd.Timestamp) -> None:
        """Обновляет дневные счетчики."""
        trade_day = bar_time.normalize()
        if self._current_trade_day != trade_day:
            self._current_trade_day = trade_day
            self._daily_sl_count = 0

    def _process_exits(self, bar_batch: dict[str, Bar]) -> None:
        """Phase 1: проверка TP/SL и закрытие позиций."""
        for symbol in self._symbols:
            position = self._portfolio.position(symbol)
            if position is None:
                continue

            bar = bar_batch.get(symbol)
            if bar is None:
                continue

            # Проверяем касание TP/SL
            exit_triggered, exit_price, reason = self._check_exit(
                position, bar
            )

            if not exit_triggered:
                continue

            # Закрываем позицию в портфолио
            pnl_clean, trade_profit, commission = (
                self._portfolio.close_position_at_price(symbol, exit_price)
            )

            # Отправляем ордер на биржу (для live)
            side = "SELL" if position["dir"] == 1 else "BUY"
            try:
                self._executor.submit_market_order(
                    symbol, side, position["size"]
                )
            except Exception as e:
                print(f"[TradingEngine] Order error: {e}")

            # Записываем сделку
            trade = TradeRecord(
                trade_number=position["trade_number"],
                symbol=symbol,
                direction="LONG" if position["dir"] == 1 else "SHORT",
                entry_time=position["ts_open"],
                exit_time=bar.timestamp,
                entry_price=position["entry"],
                exit_price=exit_price,
                size=position["size"],
                margin=position["margin"],
                pnl_pct=pnl_clean,
                pnl_abs=trade_profit,
                commission=commission,
                reason=reason,
                stop_pct=position["stop_pct"],
                take_pct=position["take_pct"],
            )
            self._trades.append(trade)

            # Обновляем статистику
            if pnl_clean > 0:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

            if reason == "SL":
                self._daily_sl_count += 1
                # Устанавливаем cooldown
                if self._sl_cooldown_bars > 0:
                    cooldown_time = bar.timestamp + pd.Timedelta(
                        minutes=self._sl_cooldown_bars
                        * self._timeframe_to_minutes()
                    )
                    self._stop_cooldown_until[symbol] = cooldown_time

            # Callback
            if self._on_trade:
                self._on_trade(trade)

            # Вывод
            pnl_sign = "+" if pnl_clean > 0 else ""
            icon = "✅" if reason == "TP" else "❌" if reason == "SL" else "⏹"
            print(
                f"[{bar.timestamp}] #{position['trade_number']} {icon} {symbol}: "
                f"{reason} | PnL: {pnl_sign}{pnl_clean*100:.2f}% | "
                f"Bal: ${self._portfolio.cash:.2f}"
            )

    def _check_exit(
        self, position: dict, bar: Bar
    ) -> tuple[bool, float, str]:
        """
        Проверяет касание TP/SL на баре.

        Returns:
            (exit_triggered, exit_price, reason)
        """
        direction = position["dir"]
        entry = position["entry"]
        stop_pct = position["stop_pct"]
        take_pct = position["take_pct"]

        if direction == 1:  # LONG
            stop_price = entry * (1 - stop_pct)
            take_price = entry * (1 + take_pct)

            # Проверяем SL first (приоритет)
            if bar.low <= stop_price:
                # Исполняем по худшей цене (open гэп или stop_price)
                fill_price = min(bar.open, stop_price)
                # Применяем slippage
                fill_price *= 1 - self._slippage
                return True, fill_price, "SL"

            # Проверяем TP
            if bar.high >= take_price:
                fill_price = max(bar.open, take_price)
                fill_price *= 1 - self._slippage
                return True, fill_price, "TP"

        else:  # SHORT
            stop_price = entry * (1 + stop_pct)
            take_price = entry * (1 - take_pct)

            # SL first
            if bar.high >= stop_price:
                fill_price = max(bar.open, stop_price)
                fill_price *= 1 + self._slippage
                return True, fill_price, "SL"

            # TP
            if bar.low <= take_price:
                fill_price = min(bar.open, take_price)
                fill_price *= 1 + self._slippage
                return True, fill_price, "TP"

        return False, 0.0, ""

    def _process_entries(self, bar_batch: dict[str, Bar]) -> None:
        """Phase 2: поиск и открытие новых позиций."""
        # Проверяем дневной лимит SL
        if self._max_sl_per_day > 0 and self._daily_sl_count >= self._max_sl_per_day:
            return

        candidates = []

        for symbol, bar in bar_batch.items():
            # Проверяем что нет позиции
            if self._portfolio.position(symbol) is not None:
                continue

            # Проверяем cooldown
            if symbol in self._stop_cooldown_until:
                if self._clock.now() < self._stop_cooldown_until[symbol]:
                    continue
                else:
                    del self._stop_cooldown_until[symbol]

            # Получаем сигнал от ML
            signal_result = self._brain.predict(symbol, bar.timestamp)
            if signal_result is None:
                continue

            signal, prob, stop_pct, take_pct = signal_result

            if signal == 0:  # Нет сигнала
                continue

            # Расчет размера позиции
            snapshot_balance = self._portfolio.cash
            position_notional, margin = (
                self._risk.raw_entry_notional_and_margin(
                    snapshot_balance, stop_pct, self._consecutive_losses
                )
            )

            if not self._risk.passes_min_notional(position_notional):
                continue

            # Проверяем доступный баланс
            if margin > self._portfolio.available_balance():
                continue

            # Определяем цену входа
            if self._use_open_price:
                entry_price = bar.open
            else:
                entry_price = bar.close

            # Применяем slippage к входу
            if signal == 1:  # LONG
                entry_price *= 1 + self._slippage
            else:  # SHORT
                entry_price *= 1 - self._slippage

            # Score для ранжирования
            score = prob * 10  # Упрощенный скоринг

            candidates.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "prob": prob,
                    "score": score,
                    "entry_price": entry_price,
                    "position_notional": position_notional,
                    "margin": margin,
                    "stop_pct": stop_pct,
                    "take_pct": take_pct,
                    "bar": bar,
                }
            )

        # Сортируем по score (убывание)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Открываем позиции
        for candidate in candidates:
            # Проверяем лимиты ExecutionService
            open_count = sum(
                1 for p in self._portfolio.state.positions.values() if p
            )

            if not self._execution.may_open_more(
                self._opened_this_bar, open_count
            ):
                break

            # Финальная проверка баланса
            if candidate["margin"] > self._portfolio.available_balance():
                continue

            # Открываем позицию
            self._open_position(candidate)
            self._opened_this_bar += 1

    def _open_position(self, candidate: dict) -> None:
        """Открывает позицию."""
        symbol = candidate["symbol"]
        signal = candidate["signal"]
        bar = candidate["bar"]

        trade_number = self._trade_counter
        self._trade_counter += 1

        # Добавляем в портфолио
        self._portfolio.open_position(
            symbol=symbol,
            trade_number=trade_number,
            direction=signal,
            entry_price=candidate["entry_price"],
            position_notional=candidate["position_notional"],
            required_margin=candidate["margin"],
            stop_pct=candidate["stop_pct"],
            take_pct=candidate["take_pct"],
            ts_open=bar.timestamp,
        )

        # Отправляем ордер на биржу
        side = "BUY" if signal == 1 else "SELL"
        try:
            self._executor.submit_market_order(
                symbol, side, candidate["position_notional"]
            )
        except Exception as e:
            print(f"[TradingEngine] Entry order error: {e}")

        # Вывод
        direction_str = "LONG" if signal == 1 else "SHORT"
        print(
            f"[{bar.timestamp}] #{trade_number} 🔥 OPEN {direction_str}: {symbol} "
            f"(prob={candidate['prob']:.2f}, score={candidate['score']:.2f}) "
            f"@ {candidate['entry_price']:.4f} | "
            f"Size: ${candidate['position_notional']:.2f} "
            f"Margin: ${candidate['margin']:.2f}"
        )

    def _finalize(self) -> None:
        """Финализация — закрытие оставшихся позиций."""
        print("\n[TradingEngine] Finalizing...")

        # Получаем финальные цены
        mark_prices = self._feed.get_mark_prices()

        for symbol in self._symbols:
            position = self._portfolio.position(symbol)
            if position is None:
                continue

            mark_price = mark_prices.get(symbol)
            if mark_price is None:
                continue

            # Закрываем по mark price с slippage
            if position["dir"] == 1:
                exit_price = mark_price * (1 - self._slippage)
                side = "SELL"
            else:
                exit_price = mark_price * (1 + self._slippage)
                side = "BUY"

            pnl_clean, trade_profit, commission = (
                self._portfolio.close_position_at_price(symbol, exit_price)
            )

            # Отправляем ордер
            try:
                self._executor.submit_market_order(
                    symbol, side, position["size"]
                )
            except Exception as e:
                print(f"[TradingEngine] Final close error: {e}")

            # Записываем
            trade = TradeRecord(
                trade_number=position["trade_number"],
                symbol=symbol,
                direction="LONG" if position["dir"] == 1 else "SHORT",
                entry_time=position["ts_open"],
                exit_time=self._clock.now(),
                entry_price=position["entry"],
                exit_price=exit_price,
                size=position["size"],
                margin=position["margin"],
                pnl_pct=pnl_clean,
                pnl_abs=trade_profit,
                commission=commission,
                reason="FINAL",
                stop_pct=position["stop_pct"],
                take_pct=position["take_pct"],
            )
            self._trades.append(trade)

            pnl_sign = "+" if pnl_clean > 0 else ""
            print(
                f"[{self._clock.now()}] #{position['trade_number']} ⏹ CLOSE {symbol}: "
                f"FINAL | PnL: {pnl_sign}{pnl_clean*100:.2f}% | "
                f"Bal: ${self._portfolio.cash:.2f}"
            )

        # Финальный equity
        final_equity = self._portfolio.cash
        print(f"\n[TradingEngine] Final balance: ${final_equity:.2f}")
        print(f"[TradingEngine] Total trades: {len(self._trades)}")
        print("=" * 60)

    def _timeframe_to_minutes(self) -> int:
        """Конвертирует timeframe в минуты."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(self._timeframe, 60)
