import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def iter_monthly_timestamp_splits(unique_ts, train_months, test_months, window_mode="expanding"):
    timestamps = pd.Series(pd.to_datetime(unique_ts, errors="coerce")).dropna().sort_values()
    if timestamps.empty:
        return
    if window_mode not in {"expanding", "rolling"}:
        raise ValueError("window_mode must be either 'expanding' or 'rolling'.")

    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    train_end = first_ts + pd.DateOffset(months=train_months)
    fold_idx = 1

    while train_end < last_ts:
        test_end = train_end + pd.DateOffset(months=test_months)
        if window_mode == "rolling":
            train_start = train_end - pd.DateOffset(months=train_months)
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
        else:
            train_mask = timestamps < train_end
        test_mask = (timestamps >= train_end) & (timestamps < test_end)
        train_timestamps = timestamps.loc[train_mask].to_numpy()
        test_timestamps = timestamps.loc[test_mask].to_numpy()

        if len(train_timestamps) > 0 and len(test_timestamps) > 0:
            yield fold_idx, train_timestamps, test_timestamps
            fold_idx += 1

        train_end = test_end


def build_timestamp_splits(
    unique_ts,
    n_splits,
    split_mode,
    monthly_train_months,
    monthly_test_months,
    monthly_window_mode="expanding",
):
    if split_mode == "monthly":
        if monthly_train_months <= 0 or monthly_test_months <= 0:
            raise ValueError("monthly train/test windows must be positive month counts.")
        return list(
            iter_monthly_timestamp_splits(
                unique_ts,
                monthly_train_months,
                monthly_test_months,
                window_mode=monthly_window_mode,
            )
        )

    n_timestamps = len(unique_ts)
    if n_timestamps < n_splits + 1:
        raise RuntimeError(
            f"Only {n_timestamps} unique timestamps — need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [
        (fold_idx, unique_ts[train_ts_idx], unique_ts[test_ts_idx])
        for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(tscv.split(unique_ts), start=1)
    ]
