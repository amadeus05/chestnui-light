from __future__ import annotations

import pandas as pd


def merge_main_and_htf_features(
    main_features: pd.DataFrame,
    htf_features: pd.DataFrame | None,
    htf_feature_columns: list[str],
    timestamp_column: str = "timestamp_ms",
) -> pd.DataFrame:
    output = main_features.copy().sort_values(timestamp_column).reset_index(drop=True)
    if htf_features is None or htf_features.empty:
        for column in htf_feature_columns:
            if column not in output.columns:
                output[column] = pd.NA
        return output

    merge_columns = [timestamp_column] + [
        column for column in htf_feature_columns if column in htf_features.columns
    ]
    merged = pd.merge_asof(
        output,
        htf_features[merge_columns].sort_values(timestamp_column).reset_index(drop=True),
        on=timestamp_column,
        direction="backward",
    )
    for column in htf_feature_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged
