from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

import bt_walk_forward
import config as cfg


SplitMode = Literal["tscv", "monthly"]
MonthlyWindowMode = Literal["expanding", "rolling"]


@dataclass(frozen=True, slots=True)
class LightGbmWalkForwardRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] | None = None
    seed: int = 42
    n_splits: int = 5
    split_mode: SplitMode = "tscv"
    monthly_train_months: int = 6
    monthly_test_months: int = 1
    monthly_window_mode: MonthlyWindowMode = "expanding"
    purge_gap: int = cfg.effective_max_label_horizon()

    def resolved_symbols(self) -> list[str]:
        return list(self.symbols or cfg.SYMBOLS)


@dataclass(slots=True)
class LightGbmWalkForwardResult:
    predictions: pd.DataFrame
    fold_details: list[dict]
    feature_columns: list[str]
    event_filter: dict
    symbols: list[str]
    request: LightGbmWalkForwardRequest


@dataclass(slots=True)
class LightGbmWalkForwardPlan:
    candidate_rows: int
    directional_rows: int
    feature_count: int
    symbols: list[str]
    split_mode: SplitMode
    monthly_window_mode: MonthlyWindowMode
    purge_gap: int
    folds: list[dict]

    @property
    def estimated_prediction_rows(self) -> int:
        return sum(int(fold["candidate_test_rows"]) for fold in self.folds)


class LightGbmWalkForwardRunner:
    """Walk-forward OOS prediction builder for the LightGBM model."""

    def plan(self, request: LightGbmWalkForwardRequest) -> LightGbmWalkForwardPlan:
        full_frame, candidate_frame, directional_frame, feature_columns, _ = (
            bt_walk_forward.load_candidate_and_training_frames(
                request.db_path,
                request.resolved_symbols(),
            )
        )
        splits = self.build_timestamp_splits(full_frame, request)
        folds = []
        for fold_idx, train_timestamps, test_timestamps in splits:
            original_train_timestamps = train_timestamps
            if request.purge_gap > 0 and len(train_timestamps) > request.purge_gap:
                train_timestamps = train_timestamps[:-request.purge_gap]

            train_ts_set = set(train_timestamps)
            test_ts_set = set(test_timestamps)
            train_rows = int(directional_frame[bt_walk_forward.train.TIMESTAMP_COLUMN].isin(train_ts_set).sum())
            candidate_test_rows = int(candidate_frame[bt_walk_forward.train.TIMESTAMP_COLUMN].isin(test_ts_set).sum())
            folds.append(
                {
                    "fold": int(fold_idx),
                    "train_timestamps": int(len(train_timestamps)),
                    "test_timestamps": int(len(test_timestamps)),
                    "train_rows": train_rows,
                    "candidate_test_rows": candidate_test_rows,
                    "train_start": str(pd.to_datetime(train_timestamps[0])) if len(train_timestamps) else None,
                    "train_end_after_purge": str(pd.to_datetime(train_timestamps[-1])) if len(train_timestamps) else None,
                    "train_end_before_purge": str(pd.to_datetime(original_train_timestamps[-1])) if len(original_train_timestamps) else None,
                    "test_start": str(pd.to_datetime(test_timestamps[0])) if len(test_timestamps) else None,
                    "test_end": str(pd.to_datetime(test_timestamps[-1])) if len(test_timestamps) else None,
                }
            )

        return LightGbmWalkForwardPlan(
            candidate_rows=int(len(candidate_frame)),
            directional_rows=int(len(directional_frame)),
            feature_count=int(len(feature_columns)),
            symbols=request.resolved_symbols(),
            split_mode=request.split_mode,
            monthly_window_mode=request.monthly_window_mode,
            purge_gap=int(request.purge_gap),
            folds=folds,
        )

    def run(self, request: LightGbmWalkForwardRequest) -> LightGbmWalkForwardResult:
        full_frame, candidate_frame, directional_frame, feature_columns, event_filter_config = (
            bt_walk_forward.load_candidate_and_training_frames(
                request.db_path,
                request.resolved_symbols(),
            )
        )

        if request.split_mode == "monthly":
            predictions, fold_details = bt_walk_forward.build_monthly_walk_forward_predictions(
                full_frame=full_frame,
                candidate_frame=candidate_frame,
                directional_frame=directional_frame,
                feature_columns=feature_columns,
                train_months=request.monthly_train_months,
                test_months=request.monthly_test_months,
                window_mode=request.monthly_window_mode,
                purge_gap=request.purge_gap,
                seed=request.seed,
            )
        else:
            predictions, fold_details = bt_walk_forward.build_walk_forward_predictions(
                full_frame=full_frame,
                candidate_frame=candidate_frame,
                directional_frame=directional_frame,
                feature_columns=feature_columns,
                n_splits=request.n_splits,
                purge_gap=request.purge_gap,
                seed=request.seed,
            )

        return LightGbmWalkForwardResult(
            predictions=predictions,
            fold_details=fold_details,
            feature_columns=list(feature_columns),
            event_filter=dict(event_filter_config),
            symbols=request.resolved_symbols(),
            request=request,
        )

    @staticmethod
    def build_timestamp_splits(full_frame: pd.DataFrame, request: LightGbmWalkForwardRequest):
        unique_ts = pd.Series(full_frame[bt_walk_forward.train.TIMESTAMP_COLUMN].dropna().unique()).sort_values().to_numpy()
        if request.split_mode == "monthly":
            return list(
                bt_walk_forward.iter_monthly_splits(
                    unique_ts,
                    request.monthly_train_months,
                    request.monthly_test_months,
                    window_mode=request.monthly_window_mode,
                )
            )

        from sklearn.model_selection import TimeSeriesSplit

        splitter = TimeSeriesSplit(n_splits=request.n_splits)
        return [
            (fold_idx, unique_ts[train_ts_idx], unique_ts[test_ts_idx])
            for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(splitter.split(unique_ts), start=1)
        ]
