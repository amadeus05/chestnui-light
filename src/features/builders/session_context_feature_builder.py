from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class SessionContextFeatureBuilder(FeatureBuilderContract):
    block_name = "session_context"

    def provides(self) -> set[str]:
        return {
            "intraday_time_sin",
            "intraday_time_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_asia_session",
            "is_london_session",
            "is_ny_session",
            "is_london_ny_overlap",
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output

        timestamps = pd.to_datetime(context.frame["timestamp"], errors="coerce")
        hour = timestamps.dt.hour.astype(float)
        minute = timestamps.dt.minute.astype(float)
        minute_of_day = hour * 60.0 + minute
        day_of_week = timestamps.dt.dayofweek.astype(float)

        intraday_angle = 2.0 * np.pi * minute_of_day / 1440.0
        day_of_week_angle = 2.0 * np.pi * day_of_week / 7.0

        # Assumption: timestamps are UTC; session buckets are defined in UTC for crypto.
        is_asia = ((hour >= 0) & (hour < 8)).astype(float)
        is_london = ((hour >= 7) & (hour < 16)).astype(float)
        is_ny = ((hour >= 13) & (hour < 22)).astype(float)
        is_overlap = ((hour >= 13) & (hour < 16)).astype(float)

        if "intraday_time_sin" in active:
            output["intraday_time_sin"] = np.sin(intraday_angle)
        if "intraday_time_cos" in active:
            output["intraday_time_cos"] = np.cos(intraday_angle)
        if "day_of_week_sin" in active:
            output["day_of_week_sin"] = np.sin(day_of_week_angle)
        if "day_of_week_cos" in active:
            output["day_of_week_cos"] = np.cos(day_of_week_angle)
        if "is_asia_session" in active:
            output["is_asia_session"] = is_asia
        if "is_london_session" in active:
            output["is_london_session"] = is_london
        if "is_ny_session" in active:
            output["is_ny_session"] = is_ny
        if "is_london_ny_overlap" in active:
            output["is_london_ny_overlap"] = is_overlap
        return output
