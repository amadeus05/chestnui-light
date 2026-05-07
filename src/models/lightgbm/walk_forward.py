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


class LightGbmWalkForwardRunner:
    """Walk-forward OOS prediction builder for the LightGBM model."""

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
