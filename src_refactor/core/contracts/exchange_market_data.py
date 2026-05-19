from __future__ import annotations

from typing import Protocol

import pandas as pd


class ExchangeMarketDataClient(Protocol):
    @property
    def exchange_code(self) -> str:
        ...

    def timeframe_ms(self, timeframe: str) -> int:
        ...

    def fetch_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        ...

    def fetch_premium_index_klines(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        ...

    def fetch_funding_rates(
        self,
        *,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        ...

    def fetch_open_interest(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        ...
