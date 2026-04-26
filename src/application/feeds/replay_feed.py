"""DataFeed для бэктеста — подает исторические бары по одному timestamp."""
from __future__ import annotations

import pandas as pd

from src.contracts import Bar, DataFeed, ReplayClock
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


def _timeframe_to_ms(timeframe: str) -> int:
    """Конвертирует timeframe в миллисекунды."""
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


class ReplayDataFeed(DataFeed):
    """
    Подает исторические бары для бэктеста.

    - Загружает все данные при инициализации
    - Находит пересечение timestamp всех символов
    - Возвращает batch по одному timestamp за вызов next_bar_batch()
    - Обновляет ReplayClock при каждом шаге
    """

    def __init__(
        self,
        clock: ReplayClock,
        repository: HistoricalKlineRepository,
    ) -> None:
        super().__init__(clock)
        self._repository = repository
        self._clock: ReplayClock = clock

        # State
        self._symbols: list[str] = []
        self._timeframe: str = ""
        self._timeframe_ms: int = 0
        self._timestamps: list[pd.Timestamp] = []
        self._current_idx: int = 0
        self._bar_data: dict[str, pd.DataFrame] = {}
        self._current_batch: dict[str, Bar] = {}

    def initialize(self, symbols: list[str], timeframe: str) -> None:
        """Загрузка исторических данных."""
        self._symbols = symbols
        self._timeframe = timeframe
        self._timeframe_ms = _timeframe_to_ms(timeframe)
        self._current_idx = 0
        self._bar_data = {}
        self._current_batch = {}

        print(f"[ReplayDataFeed] Loading data for {len(symbols)} symbols...")

        # Загрузка данных для всех символов
        all_timestamps_sets = []
        for symbol in symbols:
            df = self._repository.load_candles(symbol, timeframe)
            if df.empty:
                raise RuntimeError(f"No data for {symbol} {timeframe}")

            # Конвертация timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().sort_values("timestamp").reset_index(drop=True)
            self._bar_data[symbol] = df
            all_timestamps_sets.append(set(df["timestamp"]))
            print(f"  {symbol}: {len(df)} candles")

        # Находим пересечение timestamp (только где есть данные для всех)
        common_ts = set.intersection(*all_timestamps_sets)
        self._timestamps = sorted(common_ts)

        if len(self._timestamps) < 2:
            raise RuntimeError(
                f"Not enough common timestamps: {len(self._timestamps)}"
            )

        print(f"[ReplayDataFeed] Common timestamps: {len(self._timestamps)}")
        print(f"  From: {self._timestamps[0]}")
        print(f"  To:   {self._timestamps[-1]}")

    def next_bar_batch(self) -> dict[str, Bar] | None:
        """Возвращает следующий batch баров для всех символов."""
        if self._current_idx >= len(self._timestamps):
            return None  # Конец данных

        ts = self._timestamps[self._current_idx]

        # Обновляем время в clock
        self._clock.set(ts)

        # Собираем бары для всех символов
        batch = {}
        for symbol, df in self._bar_data.items():
            row = df[df["timestamp"] == ts]
            if row.empty:
                continue

            r = row.iloc[0]
            bar = Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                open_time=ts - pd.Timedelta(self._timeframe_ms, "ms"),
            )
            batch[symbol] = bar

        self._current_batch = batch
        self._current_idx += 1
        return batch

    def get_mark_prices(self) -> dict[str, float]:
        """Возвращает close цены текущего бара."""
        return {sym: bar.close for sym, bar in self._current_batch.items()}
