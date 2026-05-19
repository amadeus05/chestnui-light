from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src_refactor.domain.features.builders.base.derivatives import (
    attach_funding_context,
    attach_open_interest_context,
    attach_premium_index_context,
)
from src_refactor.infrastructure.feeds.historical_frame_stream import CandleRepository, HistoricalCandleFrameLoader


class FundingRateRepository(Protocol):
    def load_funding_rates(self, symbol: str) -> pd.DataFrame:
        ...


class PremiumIndexRepository(Protocol):
    def load_premium_index_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ...


class OpenInterestRepository(Protocol):
    def load_open_interest(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ...


@dataclass(frozen=True, slots=True)
class HistoricalMarketMapLoader:
    candle_loader: HistoricalCandleFrameLoader
    funding_repository: FundingRateRepository | None = None
    premium_index_repository: PremiumIndexRepository | None = None
    open_interest_repository: OpenInterestRepository | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    @classmethod
    def from_repository(
        cls,
        repository: CandleRepository,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> "HistoricalMarketMapLoader":
        return cls(
            candle_loader=HistoricalCandleFrameLoader(repository),
            funding_repository=repository if hasattr(repository, "load_funding_rates") else None,
            premium_index_repository=repository if hasattr(repository, "load_premium_index_klines") else None,
            open_interest_repository=repository if hasattr(repository, "load_open_interest") else None,
            start=start,
            end=end,
        )

    def load_base_map(self, symbols: tuple[str, ...] | list[str], timeframe: str) -> dict[str, pd.DataFrame]:
        return {
            symbol: self._filter_window(
                self._attach_market_context(symbol, self.candle_loader.load_symbol(symbol, timeframe), timeframe)
            )
            for symbol in symbols
        }

    def load_htf_map(self, symbols: tuple[str, ...] | list[str], timeframe: str) -> dict[str, pd.DataFrame]:
        return {
            symbol: self._filter_window(self.candle_loader.load_symbol(symbol, timeframe))
            for symbol in symbols
        }

    def _attach_market_context(self, symbol: str, frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if self.funding_repository is not None:
            frame = attach_funding_context(frame, self.funding_repository.load_funding_rates(symbol))
        if self.premium_index_repository is not None:
            frame = attach_premium_index_context(
                frame,
                self.premium_index_repository.load_premium_index_klines(symbol, timeframe),
            )
        if self.open_interest_repository is not None:
            frame = attach_open_interest_context(
                frame,
                self.open_interest_repository.load_open_interest(symbol, timeframe),
            )
        return frame

    def _filter_window(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or (self.start is None and self.end is None):
            return frame
        output = frame.copy()
        timestamps = pd.to_datetime(output["timestamp"], errors="coerce")
        mask = timestamps.notna()
        if self.start is not None:
            mask &= timestamps >= pd.to_datetime(self.start)
        if self.end is not None:
            mask &= timestamps <= pd.to_datetime(self.end)
        return output.loc[mask].reset_index(drop=True)
