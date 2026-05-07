from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import torch

import config as cfg
import train
from lstm_candles import config_lstm_candles as candle_cfg
from lstm_candles.train_lstm_candles_walk_forward import walk_forward_lstm
from src.models.lstm.dataset import SequenceDataset
from src.models.lstm_candles.data_builder import LstmCandlesDataBuilder, LstmCandlesDataRequest


SplitMode = Literal["tscv", "monthly"]
MonthlyWindowMode = Literal["expanding", "rolling"]


@dataclass(frozen=True, slots=True)
class LstmCandlesWalkForwardRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))
    seed: int = 42
    n_splits: int = 5
    split_mode: SplitMode = "monthly"
    monthly_train_months: int = 6
    monthly_test_months: int = 1
    monthly_window_mode: MonthlyWindowMode = "expanding"
    purge_gap: int = cfg.effective_max_label_horizon()
    sequence_length: int = candle_cfg.SEQUENCE_LENGTH
    batch_size: int = candle_cfg.BATCH_SIZE
    epochs: int = candle_cfg.EPOCHS
    hidden_size: int = candle_cfg.HIDDEN_SIZE
    num_layers: int = candle_cfg.NUM_LAYERS
    dropout: float = candle_cfg.DROPOUT
    lr: float = candle_cfg.LEARNING_RATE
    weight_decay: float = candle_cfg.WEIGHT_DECAY
    patience: int = candle_cfg.EARLY_STOPPING_PATIENCE
    min_epochs_before_early_stop: int = candle_cfg.MIN_EPOCHS_BEFORE_EARLY_STOP
    early_stopping_min_delta: float = candle_cfg.EARLY_STOPPING_MIN_DELTA
    max_folds: int | None = None


@dataclass(slots=True)
class LstmCandlesWalkForwardResult:
    metrics: dict
    fold_details: list[dict]
    predictions: object
    model_state: dict | None
    feature_columns: list[str]
    symbols: list[str]
    request: LstmCandlesWalkForwardRequest


@dataclass(slots=True)
class LstmCandlesWalkForwardPlan:
    sample_rows: int
    candle_rows: int
    feature_count: int
    symbols: list[str]
    split_mode: SplitMode
    monthly_window_mode: MonthlyWindowMode
    purge_gap: int
    sequence_length: int
    folds: list[dict]

    @property
    def estimated_prediction_rows(self) -> int:
        return sum(int(fold["test_sequences"]) for fold in self.folds)


class LstmCandlesWalkForwardRunner:
    """Walk-forward OOS runner for candle-sequence LSTM."""

    def __init__(self, data_builder: LstmCandlesDataBuilder | None = None):
        self.data_builder = data_builder or LstmCandlesDataBuilder()

    def plan(self, request: LstmCandlesWalkForwardRequest) -> LstmCandlesWalkForwardPlan:
        data = self.data_builder.load(
            LstmCandlesDataRequest(
                db_path=request.db_path,
                symbols=request.symbols,
            )
        )
        timestamp_splits = self.build_timestamp_splits(data.sample_frame, request)
        folds = []
        for fold_idx, original_train_timestamps, test_timestamps in timestamp_splits:
            train_timestamps = original_train_timestamps
            if request.purge_gap > 0 and len(train_timestamps) > request.purge_gap:
                train_timestamps = train_timestamps[:-request.purge_gap]

            train_df = data.sample_frame.loc[data.sample_frame[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))].copy()
            test_df = data.sample_frame.loc[data.sample_frame[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))].copy()
            train_sequences = len(
                SequenceDataset(train_df, data.history_by_symbol, data.feature_columns, request.sequence_length)
            )
            test_sequences = len(
                SequenceDataset(test_df, data.history_by_symbol, data.feature_columns, request.sequence_length)
            )
            folds.append(
                {
                    "fold": int(fold_idx),
                    "train_timestamps": int(len(train_timestamps)),
                    "test_timestamps": int(len(test_timestamps)),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "train_sequences": int(train_sequences),
                    "test_sequences": int(test_sequences),
                    "train_start": str(pd.to_datetime(train_timestamps[0])) if len(train_timestamps) else None,
                    "train_end_after_purge": str(pd.to_datetime(train_timestamps[-1])) if len(train_timestamps) else None,
                    "train_end_before_purge": str(pd.to_datetime(original_train_timestamps[-1]))
                    if len(original_train_timestamps)
                    else None,
                    "test_start": str(pd.to_datetime(test_timestamps[0])) if len(test_timestamps) else None,
                    "test_end": str(pd.to_datetime(test_timestamps[-1])) if len(test_timestamps) else None,
                }
            )

        return LstmCandlesWalkForwardPlan(
            sample_rows=int(len(data.sample_frame)),
            candle_rows=int(len(data.candle_frame)),
            feature_count=int(len(data.feature_columns)),
            symbols=list(request.symbols),
            split_mode=request.split_mode,
            monthly_window_mode=request.monthly_window_mode,
            purge_gap=int(request.purge_gap),
            sequence_length=int(request.sequence_length),
            folds=folds,
        )

    def run(
        self,
        request: LstmCandlesWalkForwardRequest,
        device: torch.device | None = None,
    ) -> LstmCandlesWalkForwardResult:
        data = self.data_builder.load(
            LstmCandlesDataRequest(
                db_path=request.db_path,
                symbols=request.symbols,
            )
        )
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = self.to_legacy_args(request)
        metrics, fold_details, predictions, model_state = walk_forward_lstm(
            sample_frame=data.sample_frame,
            history_by_symbol=data.history_by_symbol,
            feature_columns=data.feature_columns,
            args=args,
            device=device,
        )
        return LstmCandlesWalkForwardResult(
            metrics=metrics,
            fold_details=fold_details,
            predictions=predictions,
            model_state=model_state,
            feature_columns=list(data.feature_columns),
            symbols=list(request.symbols),
            request=request,
        )

    @staticmethod
    def build_timestamp_splits(sample_frame, request: LstmCandlesWalkForwardRequest):
        unique_ts = sample_frame.attrs.get(
            "all_timestamps",
            pd.Series(sample_frame[train.TIMESTAMP_COLUMN].dropna().unique()).sort_values().to_numpy(),
        )
        timestamp_splits = train.build_timestamp_splits(
            unique_ts=unique_ts,
            n_splits=request.n_splits,
            split_mode=request.split_mode,
            monthly_train_months=request.monthly_train_months,
            monthly_test_months=request.monthly_test_months,
            monthly_window_mode=request.monthly_window_mode,
        )
        if request.max_folds is not None:
            timestamp_splits = timestamp_splits[: request.max_folds]
        return timestamp_splits

    @staticmethod
    def to_legacy_args(request: LstmCandlesWalkForwardRequest) -> Namespace:
        return Namespace(
            db_path=request.db_path,
            symbols=request.symbols,
            seed=request.seed,
            n_splits=request.n_splits,
            split_mode=request.split_mode,
            monthly_train_months=request.monthly_train_months,
            monthly_test_months=request.monthly_test_months,
            monthly_window_mode=request.monthly_window_mode,
            purge_gap=request.purge_gap,
            sequence_length=request.sequence_length,
            batch_size=request.batch_size,
            epochs=request.epochs,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            dropout=request.dropout,
            lr=request.lr,
            weight_decay=request.weight_decay,
            patience=request.patience,
            min_epochs_before_early_stop=request.min_epochs_before_early_stop,
            early_stopping_min_delta=request.early_stopping_min_delta,
            max_folds=request.max_folds,
        )
