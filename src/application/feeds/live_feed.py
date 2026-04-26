"""DataFeed для paper/live trading — получает данные через polling или websocket."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

from src.contracts import Bar, DataFeed, WallClock

if TYPE_CHECKING:
    from src.contracts.exchange_contract import ExchangeContract


def _timeframe_to_seconds(timeframe: str) -> int:
    """Конвертирует timeframe в секунды для sleep."""
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


class LiveDataFeed(DataFeed):
    """
    Получает live данные через polling или websocket.

    Режим работы:
    1. Ждем закрытия бара (по времени)
    2. Получаем завершенный бар для всех символов
    3. Возвращает batch

    Для paper/live режимов.
    """

    def __init__(
        self,
        clock: WallClock,
        exchange: ExchangeContract,
        poll_interval_sec: float = 5.0,
        max_wait_sec: float = 30.0,
    ) -> None:
        super().__init__(clock)
        self._exchange = exchange
        self._clock: WallClock = clock
        self._poll_interval = poll_interval_sec
        self._max_wait = max_wait_sec

        # State
        self._symbols: list[str] = []
        self._timeframe: str = ""
        self._timeframe_sec: int = 0
        self._latest_closed_bars: dict[str, Bar] = {}
        self._pending_bar: dict[str, Bar] = {}  # Текущий незакрытый бар
        self._last_batch_time: pd.Timestamp | None = None
        self._is_running: bool = False

    def initialize(self, symbols: list[str], timeframe: str) -> None:
        """Инициализация: подписка на потоки или первый fetch."""
        self._symbols = symbols
        self._timeframe = timeframe
        self._timeframe_sec = _timeframe_to_seconds(timeframe)
        self._latest_closed_bars = {}
        self._pending_bar = {}
        self._last_batch_time = None
        self._is_running = True

        print(f"[LiveDataFeed] Initialized for {len(symbols)} symbols, {timeframe}")
        print(f"  Poll interval: {self._poll_interval}s")

        # Предзагрузка последних баров
        self._preload_latest_bars()

    def _preload_latest_bars(self) -> None:
        """Загрузка последних N баров для инициализации."""
        now = self._clock.now()
        lookback = pd.Timedelta(minutes=10)  # Загружаем последние 10 минут
        start_ts = int((now - lookback).timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)

        for symbol in self._symbols:
            try:
                klines = self._exchange.fetch_klines(
                    symbol, self._timeframe, start_ts, end_ts
                )
                if klines:
                    last = klines[-1]
                    bar = Bar(
                        symbol=symbol,
                        timestamp=pd.Timestamp(last.close_time, unit="ms"),
                        open=last.open,
                        high=last.high,
                        low=last.low,
                        close=last.close,
                        volume=last.volume,
                        open_time=pd.Timestamp(last.open_time, unit="ms"),
                    )
                    self._latest_closed_bars[symbol] = bar
                    print(f"  {symbol}: preloaded bar @ {bar.timestamp}")
            except Exception as e:
                print(f"  Warning: failed to preload {symbol}: {e}")

    def next_bar_batch(self) -> dict[str, Bar] | None:
        """
        Ждет и возвращает следующий batch закрытых баров.

        Алгоритм:
        1. Определяем время следующего закрытия бара
        2. Ждем пока наступит это время + небольшой buffer
        3. Получаем бары для всех символов
        4. Проверяем что timestamp новый
        5. Возвращаем batch
        """
        if not self._is_running:
            return None

        now = self._clock.now()

        # Расчет времени следующего закрытия бара
        if self._last_batch_time is None:
            # Первый запуск — ждем следующее закрытие
            seconds_since_epoch = int(now.timestamp())
            next_close_seconds = (
                (seconds_since_epoch // self._timeframe_sec + 1)
                * self._timeframe_sec
            )
            wait_seconds = next_close_seconds - seconds_since_epoch
        else:
            # Обычный запуск — ждем timeframe секунд от последнего
            next_close = self._last_batch_time + pd.Timedelta(
                seconds=self._timeframe_sec
            )
            wait_seconds = (next_close - now).total_seconds()

        if wait_seconds > 0:
            print(
                f"[LiveDataFeed] Waiting {wait_seconds:.1f}s for next bar close..."
            )
            time.sleep(wait_seconds)

        # Получаем новые бары с retry
        batch = self._fetch_bar_batch_with_retry()

        if batch:
            self._last_batch_time = self._clock.now()
            self._latest_closed_bars = batch

        return batch if batch else None

    def _fetch_bar_batch_with_retry(self) -> dict[str, Bar] | None:
        """Получает бары для всех символов с retry логикой."""
        batch: dict[str, Bar] = {}
        deadline = time.time() + self._max_wait

        while time.time() < deadline and len(batch) < len(self._symbols):
            for symbol in self._symbols:
                if symbol in batch:
                    continue

                try:
                    now = self._clock.now()
                    # Запрашиваем последние 2 бара
                    start_ts = int(
                        (now - pd.Timedelta(minutes=5)).timestamp() * 1000
                    )
                    end_ts = int(now.timestamp() * 1000)

                    klines = self._exchange.fetch_klines(
                        symbol, self._timeframe, start_ts, end_ts
                    )

                    if not klines:
                        continue

                    # Берем последний закрытый бар
                    last_kline = klines[-1]
                    timestamp = pd.Timestamp(last_kline.close_time, unit="ms")

                    # Проверяем что бар новый (не дубль)
                    if symbol in self._latest_closed_bars:
                        last_ts = self._latest_closed_bars[symbol].timestamp
                        if timestamp <= last_ts:
                            continue  # Еще старый бар

                    bar = Bar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=last_kline.open,
                        high=last_kline.high,
                        low=last_kline.low,
                        close=last_kline.close,
                        volume=last_kline.volume,
                        open_time=pd.Timestamp(last_kline.open_time, unit="ms"),
                    )
                    batch[symbol] = bar
                    print(f"[LiveDataFeed] {symbol} bar @ {timestamp}")

                except Exception as e:
                    print(f"[LiveDataFeed] Error fetching {symbol}: {e}")

            if len(batch) < len(self._symbols):
                time.sleep(self._poll_interval)

        return batch if batch else None

    def get_mark_prices(self) -> dict[str, float]:
        """
        Возвращает текущие mark prices.

        Для live: использует close последнего полученного бара.
        """
        return {
            sym: bar.close for sym, bar in self._latest_closed_bars.items()
        }

    def stop(self) -> None:
        """Остановка фида."""
        self._is_running = False
