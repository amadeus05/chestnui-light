import numpy as np
import pandas as pd

from .constants import SYMBOL_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN
from .dataset import build_timestamp_profile

def build_period_payload(frame):
    if frame.empty:
        return None
    return {
        "start": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "end": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


def build_fold_boundary_summary(fold_details):
    return [
        {
            "fold": int(fold["fold"]),
            "train_rows": int(fold["train_rows"]),
            "test_rows": int(fold["test_rows"]),
            **fold.get("timestamp_boundaries", {}),
        }
        for fold in fold_details
    ]


def build_dataset_diagnostics(dataset, fold_details):
    all_timestamp_profile = dataset.attrs.get("all_timestamps_profile")
    if not all_timestamp_profile:
        all_timestamp_profile = build_timestamp_profile(dataset.attrs.get("all_timestamps", []))

    candidate_rows_before_filter = int(dataset.attrs.get("candidate_rows_before_filter", len(dataset)))
    candidate_rows_after_filter = int(dataset.attrs.get("candidate_rows", len(dataset)))
    excluded_by_event_filter_rows = int(dataset.attrs.get("excluded_by_event_filter_rows", 0))
    excluded_non_directional_rows = int(dataset.attrs.get("excluded_non_directional_rows", 0))

    return {
        "all_timestamps_count": int(all_timestamp_profile["count"]),
        "first_all_timestamp": all_timestamp_profile["first"],
        "last_all_timestamp": all_timestamp_profile["last"],
        "feature_rows_after_required_columns": int(dataset.attrs.get("required_non_null_rows", 0)),
        "candidate_rows_before_filter": candidate_rows_before_filter,
        "candidate_rows_after_filter": candidate_rows_after_filter,
        "excluded_by_event_filter_rows": excluded_by_event_filter_rows,
        "directional_rows_after_filter": int(len(dataset)),
        "excluded_non_directional_rows": excluded_non_directional_rows,
        "dataset_period_after_filters": build_period_payload(dataset),
        "fold_boundary_timestamps": build_fold_boundary_summary(fold_details),
        "feature_table_row_counts_by_symbol": dataset.attrs.get("feature_table_row_counts_by_symbol", {}),
        "rows_before_filter_by_symbol": dataset.attrs.get("rows_before_filter_by_symbol", {}),
        "candidate_rows_by_symbol": dataset.attrs.get("candidate_rows_by_symbol", {}),
        "directional_rows_by_symbol": dataset.attrs.get("directional_rows_by_symbol", {}),
    }


def build_fold_stability_payload(fold_details):
    if not fold_details:
        return {
            "accuracy_std": None,
            "accuracy_range": None,
            "roc_auc_std": None,
            "roc_auc_range": None,
        }

    accuracy_values = np.asarray([float(fold["accuracy"]) for fold in fold_details], dtype=float)
    roc_auc_values = np.asarray(
        [float(fold["roc_auc"]) for fold in fold_details if fold.get("roc_auc") is not None],
        dtype=float,
    )
    return {
        "accuracy_std": float(np.std(accuracy_values)),
        "accuracy_range": float(np.max(accuracy_values) - np.min(accuracy_values)),
        "roc_auc_std": float(np.std(roc_auc_values)) if len(roc_auc_values) else None,
        "roc_auc_range": (
            float(np.max(roc_auc_values) - np.min(roc_auc_values))
            if len(roc_auc_values)
            else None
        ),
    }
