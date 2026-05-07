from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config as cfg
import train
from src.models.lstm.dataset import build_history_by_symbol
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


@dataclass(frozen=True, slots=True)
class LstmDataRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))


@dataclass(slots=True)
class LstmDataBundle:
    sample_frame: pd.DataFrame
    full_frame: pd.DataFrame
    history_by_symbol: dict[str, pd.DataFrame]
    feature_columns: list[str]


class LstmDataBuilder:
    """Build supervised samples and per-symbol feature history for ETL-sequence LSTM."""

    def load(self, request: LstmDataRequest) -> LstmDataBundle:
        sample_frame = train.load_training_frame(request.db_path, request.symbols)
        feature_columns = self.select_feature_columns(sample_frame)
        full_frame = self.load_full_feature_frame(request.db_path, request.symbols, feature_columns)
        history_by_symbol = build_history_by_symbol(full_frame, feature_columns)
        return LstmDataBundle(
            sample_frame=sample_frame,
            full_frame=full_frame,
            history_by_symbol=history_by_symbol,
            feature_columns=feature_columns,
        )

    def select_feature_columns(self, sample_frame: pd.DataFrame) -> list[str]:
        feature_columns = train.select_feature_columns(sample_frame)
        return [
            column
            for column in feature_columns
            if column != train.SYMBOL_COLUMN and pd.api.types.is_numeric_dtype(sample_frame[column])
        ]

    def load_full_feature_frame(
        self,
        db_path: str,
        symbols: list[str],
        feature_columns: list[str],
    ) -> pd.DataFrame:
        repository = HistoricalKlineRepository(db_path=db_path)
        full_frame = repository.load_feature_dataset(symbols)
        full_frame = full_frame.dropna(subset=[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN]).copy()
        end_cutoff = train.get_end_date_cutoff()
        if end_cutoff is not None and not pd.isna(end_cutoff):
            full_frame = full_frame.loc[full_frame[train.TIMESTAMP_COLUMN] <= end_cutoff].copy()

        missing = [column for column in feature_columns if column not in full_frame.columns]
        if missing:
            raise RuntimeError(f"Full feature frame is missing LSTM feature columns: {missing[:10]}")

        return full_frame
