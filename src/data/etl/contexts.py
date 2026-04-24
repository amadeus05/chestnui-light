import numpy as np
import pandas as pd


def attach_funding_context(
    base_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if funding_df is None or funding_df.empty:
        output["funding_rate"] = np.nan
        return output

    funding_frame = funding_df[["timestamp", "funding_rate"]].copy().sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        output,
        funding_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_premium_index_context(
    base_df: pd.DataFrame,
    premium_index_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if premium_index_df is None or premium_index_df.empty:
        output["premium_index_close"] = np.nan
        return output

    premium_frame = (
        premium_index_df[["timestamp", "premium_index_close"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        premium_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_open_interest_context(
    base_df: pd.DataFrame,
    open_interest_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if open_interest_df is None or open_interest_df.empty:
        output["open_interest"] = np.nan
        return output

    open_interest_frame = (
        open_interest_df[["timestamp", "open_interest"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        open_interest_frame,
        on="timestamp",
        direction="backward",
    )
    return merged
