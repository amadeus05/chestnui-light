from __future__ import annotations

import pandas as pd


def canonical_prediction_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid prediction timestamp: {value!r}")
    return timestamp.tz_convert(None)
