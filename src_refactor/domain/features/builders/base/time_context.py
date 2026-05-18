from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd

TIME_CONTEXT_FEATURE_COLUMNS = ["hour_sin_1h", "hour_cos_1h", "is_weekend_1h"]


def build_time_context_values(timestamp_ms: int) -> Mapping[str, float]:
    timestamp = pd.to_datetime(int(timestamp_ms), unit="ms", utc=True)
    hour = int(timestamp.hour)
    return {
        "hour_sin_1h": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos_1h": math.cos(2.0 * math.pi * hour / 24.0),
        "is_weekend_1h": float(timestamp.dayofweek >= 5),
    }


def build_time_context_frame(timestamp_ms: Iterable[int]) -> pd.DataFrame:
    rows = [
        {"timestamp_ms": int(value), **build_time_context_values(int(value))}
        for value in timestamp_ms
    ]
    return pd.DataFrame(rows, columns=["timestamp_ms", *TIME_CONTEXT_FEATURE_COLUMNS])
