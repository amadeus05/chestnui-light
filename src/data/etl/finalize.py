import numpy as np
import pandas as pd

from .barriers import compute_effective_horizons, get_base_horizon
from .constants import BARRIER_OUTPUT_COLUMNS, BASE_OUTPUT_COLUMNS


def finalize_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    effective_horizons = compute_effective_horizons(output)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else get_base_horizon()
    if max_horizon > 0:
        if len(output) <= max_horizon:
            empty_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
            return output.iloc[0:0][empty_columns].copy()
        output = output.iloc[:-max_horizon].copy()

    output_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    output.dropna(inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output
