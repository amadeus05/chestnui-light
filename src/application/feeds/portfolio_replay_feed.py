"""Replay по шагам «закрытие t → исполнение на баре t+1» для портфельного бэктеста."""
from __future__ import annotations

import pandas as pd

from src.contracts import Bar, ReplayClock
from src.contracts.replay_step import PortfolioReplayStep


def _tf_ms(timeframe: str) -> int:
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


class PortfolioReplayFeed:
    """
    Итерирует пары (test_timestamps[i], test_timestamps[i+1]) по подготовленным all_raw и индексам.
    БД не загружает — передайте результат подготовки данных (см. backtest_data / portfolio_backtest_runner).
    """

    def __init__(self, clock: ReplayClock, timeframe: str) -> None:
        self._clock = clock
        self._tf_ms = _tf_ms(timeframe)
        self._timestamps: list[pd.Timestamp] = []
        self._idx = 0
        self._all_raw: dict = {}
        self._main_index: dict[str, dict] = {}

    def prepare(
        self,
        *,
        all_raw: dict,
        all_main_index: dict[str, dict],
        test_timestamps: list[pd.Timestamp],
    ) -> None:
        self._all_raw = all_raw
        self._main_index = all_main_index
        self._timestamps = list(test_timestamps)
        self._idx = 0

    def next_step(self) -> PortfolioReplayStep | None:
        if self._idx >= len(self._timestamps) - 1:
            return None
        current_ts = self._timestamps[self._idx]
        next_ts = self._timestamps[self._idx + 1]
        self._idx += 1
        self._clock.set(current_ts)

        mark_prices: dict[str, float] = {}
        exec_bars: dict[str, Bar] = {}

        from src.application.backtest import backtest_data as ld

        for sym, payload in self._all_raw.items():
            curr = ld.get_exec_row_by_ts_index(self._main_index[sym], current_ts)
            nxt = ld.get_exec_row_by_ts_index(self._main_index[sym], next_ts)
            if curr is None or nxt is None:
                continue
            mark_prices[sym] = float(curr["close"])
            r = nxt
            exec_bars[sym] = Bar(
                symbol=sym,
                timestamp=next_ts,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                open_time=next_ts - pd.Timedelta(self._tf_ms, unit="ms"),
            )
        return PortfolioReplayStep(
            decision_ts=current_ts,
            exec_ts=next_ts,
            mark_prices=mark_prices,
            exec_bars=exec_bars,
        )

    def last_timestamp(self) -> pd.Timestamp | None:
        return self._timestamps[-1] if self._timestamps else None

    def final_mark_prices(self) -> dict[str, float]:
        """Close последнего бара в test_timestamps (для принудительного закрытия позиций)."""
        last = self.last_timestamp()
        out: dict[str, float] = {}
        if last is None:
            return out
        from src.application.backtest import backtest_data as ld

        for sym in self._all_raw:
            row = ld.get_exec_row_by_ts_index(self._main_index.get(sym, {}), last)
            if row is not None:
                out[sym] = float(row["close"])
        return out
