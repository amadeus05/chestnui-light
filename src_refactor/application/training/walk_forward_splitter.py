from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import WalkForwardFold


@dataclass(frozen=True, slots=True)
class WalkForwardSplitter:
    config: ExperimentConfig

    def split(self, frame: pd.DataFrame) -> list[WalkForwardFold]:
        timestamps = self._timestamps(frame)
        if timestamps.empty:
            return []

        if self.config.split_mode == "tscv":
            return self._split_tscv(timestamps)
        if self.config.split_mode == "monthly_expanding":
            return self._split_monthly(timestamps, rolling=False)
        if self.config.split_mode == "monthly_rolling":
            return self._split_monthly(timestamps, rolling=True)

        raise ValueError(f"Unsupported split mode: {self.config.split_mode}")

    def _timestamps(self, frame: pd.DataFrame) -> pd.Series:
        column = self.config.timestamp_column
        if column not in frame.columns:
            raise ValueError(f"DataFrame must contain timestamp column '{column}'.")
        return pd.to_datetime(frame[column], errors="coerce").dropna().sort_values().drop_duplicates()

    def _split_tscv(self, timestamps: pd.Series) -> list[WalkForwardFold]:
        unique_ts = timestamps.reset_index(drop=True)
        n_splits = max(1, int(self.config.n_splits))
        if len(unique_ts) < n_splits + 1:
            return []

        test_size = len(unique_ts) // (n_splits + 1)
        folds: list[WalkForwardFold] = []
        for fold_id in range(n_splits):
            test_start_idx = len(unique_ts) - test_size * (n_splits - fold_id)
            test_end_idx = test_start_idx + test_size - 1
            if test_start_idx <= 0 or test_end_idx >= len(unique_ts):
                continue

            train_end_idx = test_start_idx - self.config.purge_gap - 1
            if train_end_idx < 0:
                continue
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=unique_ts.iloc[0],
                    train_end=unique_ts.iloc[train_end_idx],
                    test_start=unique_ts.iloc[test_start_idx],
                    test_end=unique_ts.iloc[test_end_idx],
                    purge_gap=self.config.purge_gap,
                )
            )
        return folds

    def _split_monthly(self, timestamps: pd.Series, *, rolling: bool) -> list[WalkForwardFold]:
        unique_ts = timestamps.reset_index(drop=True)
        start_month = unique_ts.iloc[0].to_period("M").to_timestamp()
        end_month = unique_ts.iloc[-1].to_period("M").to_timestamp()
        month_starts = pd.date_range(start=start_month, end=end_month, freq="MS")

        train_months = max(1, int(self.config.train_months))
        test_months = max(1, int(self.config.test_months))
        folds: list[WalkForwardFold] = []
        fold_id = 0

        for test_start_month in month_starts[train_months:]:
            test_end_exclusive = test_start_month + pd.DateOffset(months=test_months)
            train_start_month = (
                test_start_month - pd.DateOffset(months=train_months)
                if rolling
                else month_starts[0]
            )
            train_end = test_start_month - pd.Timedelta(nanoseconds=1)
            if self.config.purge_gap > 0:
                train_candidates = unique_ts[unique_ts < test_start_month]
                purge_idx = len(train_candidates) - self.config.purge_gap - 1
                if purge_idx < 0:
                    continue
                train_end = min(train_end, train_candidates.iloc[purge_idx])

            train_mask = unique_ts.between(train_start_month, train_end, inclusive="both")
            test_mask = unique_ts.between(test_start_month, test_end_exclusive, inclusive="left")
            if not train_mask.any() or not test_mask.any():
                continue

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=unique_ts.loc[train_mask].iloc[0],
                    train_end=unique_ts.loc[train_mask].iloc[-1],
                    test_start=unique_ts.loc[test_mask].iloc[0],
                    test_end=unique_ts.loc[test_mask].iloc[-1],
                    purge_gap=self.config.purge_gap,
                )
            )
            fold_id += 1

        return folds[: self.config.n_splits] if self.config.n_splits > 0 else folds
