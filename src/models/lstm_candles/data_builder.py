from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config as cfg
import train
from lstm_candles.data import load_candle_sequence_frames
from src.models.lstm.dataset import build_history_by_symbol


@dataclass(frozen=True, slots=True)
class LstmCandlesDataRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))


@dataclass(slots=True)
class LstmCandlesDataBundle:
    sample_frame: pd.DataFrame
    candle_frame: pd.DataFrame
    history_by_symbol: dict[str, pd.DataFrame]
    feature_columns: list[str]


class LstmCandlesDataBuilder:
    """Build supervised labels and candle-sequence history for candle LSTM."""

    def load(self, request: LstmCandlesDataRequest) -> LstmCandlesDataBundle:
        sample_frame = train.load_training_frame(request.db_path, request.symbols)
        candle_frame, feature_columns = load_candle_sequence_frames(request.db_path, request.symbols)
        history_by_symbol = build_history_by_symbol(candle_frame, feature_columns)
        return LstmCandlesDataBundle(
            sample_frame=sample_frame,
            candle_frame=candle_frame,
            history_by_symbol=history_by_symbol,
            feature_columns=feature_columns,
        )
