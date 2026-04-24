import numpy as np
import pandas as pd
import config as cfg
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

from .config_snapshot import get_end_date_cutoff
from .constants import SYMBOL_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN
from .log import logger

def format_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return str(timestamp)


def build_timestamp_profile(values):
    timestamps = pd.Series(pd.to_datetime(values, errors="coerce")).dropna()
    if timestamps.empty:
        return {
            "count": 0,
            "first": None,
            "last": None,
        }
    return {
        "count": int(timestamps.nunique()),
        "first": format_timestamp(timestamps.min()),
        "last": format_timestamp(timestamps.max()),
    }


def build_symbol_row_profile(frame):
    if frame.empty or SYMBOL_COLUMN not in frame.columns:
        return {}

    profile = {}
    for symbol, symbol_frame in frame.groupby(SYMBOL_COLUMN, observed=True):
        symbol_key = str(symbol)
        row = {"rows": int(len(symbol_frame))}
        if TIMESTAMP_COLUMN in symbol_frame.columns:
            timestamp_profile = build_timestamp_profile(symbol_frame[TIMESTAMP_COLUMN])
            row.update(
                {
                    "unique_timestamps": timestamp_profile["count"],
                    "first_timestamp": timestamp_profile["first"],
                    "last_timestamp": timestamp_profile["last"],
                }
            )
        if TARGET_COLUMN in symbol_frame.columns:
            target_counts = symbol_frame[TARGET_COLUMN].value_counts(dropna=False).sort_index()
            row["target_counts"] = {str(key): int(value) for key, value in target_counts.items()}
        profile[symbol_key] = row
    return profile


def load_training_frame(db_path, symbols):
    """Load dataset, filter events, keep only directional labels {-1, 1} → {0, 1}."""
    repository = HistoricalKlineRepository(db_path=db_path)
    dataset = repository.load_feature_dataset(symbols)
    feature_table_row_counts_by_symbol = build_symbol_row_profile(dataset)

    dataset = dataset.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_non_null_rows = int(len(dataset))

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is not None and not pd.isna(end_cutoff):
        before_rows = len(dataset)
        dataset = dataset.loc[dataset[TIMESTAMP_COLUMN] <= end_cutoff].copy()
        logger.info(
            "Applied END_DATE cutoff at %s: kept %s/%s rows",
            end_cutoff,
            len(dataset),
            before_rows,
        )

    rows_before_event_filter = int(len(dataset))
    rows_before_filter_by_symbol = build_symbol_row_profile(dataset)

    raw_labels = dataset[TARGET_COLUMN].astype(int)
    unknown_labels = sorted(set(raw_labels.unique()) - {-1, 0, 1})
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {TARGET_COLUMN}: {unknown_labels}")

    all_timestamps = np.sort(dataset[TIMESTAMP_COLUMN].dropna().unique())

    event_filter_config = resolve_event_filter_config()
    candidate_mask = build_candidate_event_mask(dataset, event_filter_config)
    dataset.attrs["event_filter_config"] = event_filter_config
    dataset.attrs["candidate_rows"] = int(candidate_mask.sum())
    dataset.attrs["candidate_rows_before_filter"] = rows_before_event_filter
    dataset.attrs["excluded_by_event_filter_rows"] = int((~candidate_mask).sum())
    candidate_rows_by_symbol = build_symbol_row_profile(dataset.loc[candidate_mask])
    dataset.attrs["candidate_rows_by_symbol"] = candidate_rows_by_symbol
    dataset = dataset.loc[candidate_mask].copy()

    directional_mask = dataset[TARGET_COLUMN].astype(int) != 0
    excluded_non_directional_rows = int((~directional_mask).sum())
    directional_rows_by_symbol = build_symbol_row_profile(dataset.loc[directional_mask])
    dataset = dataset.loc[directional_mask].copy()
    raw_directional_labels = dataset[TARGET_COLUMN].astype(int)
    dataset[TARGET_COLUMN] = raw_directional_labels.map({-1: 0, 1: 1})
    dataset[SYMBOL_COLUMN] = dataset[SYMBOL_COLUMN].astype("category")
    dataset.attrs["excluded_non_directional_rows"] = excluded_non_directional_rows
    dataset.attrs["all_timestamps"] = all_timestamps
    dataset.attrs["all_timestamps_profile"] = build_timestamp_profile(all_timestamps)
    dataset.attrs["required_non_null_rows"] = required_non_null_rows
    dataset.attrs["rows_before_filter_by_symbol"] = rows_before_filter_by_symbol
    dataset.attrs["candidate_rows_by_symbol"] = candidate_rows_by_symbol
    dataset.attrs["directional_rows_by_symbol"] = directional_rows_by_symbol
    dataset.attrs["feature_table_row_counts_by_symbol"] = feature_table_row_counts_by_symbol
    return dataset
