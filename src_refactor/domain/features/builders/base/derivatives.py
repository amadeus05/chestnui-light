from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FUNDING_FEATURE_COLUMNS = [
    "funding_rate_8h",
    "funding_rate_zscore_7d",
    "funding_rate_change_24h",
    "longs_overheated_1h",
    "shorts_overheated_1h",
]
PREMIUM_FEATURE_COLUMNS = [
    "premium_index_1h",
    "premium_index_zscore_7d",
    "premium_index_change_24h",
    "crowded_longs_score_1h",
    "crowded_shorts_score_1h",
]
OPEN_INTEREST_FEATURE_COLUMNS = [
    "open_interest_zscore_7d",
    "open_interest_change_pct_8h",
    "open_interest_change_pct_24h",
    "open_interest_funding_crowding_1h",
]
DERIVATIVES_FEATURE_COLUMNS = [
    *FUNDING_FEATURE_COLUMNS,
    *PREMIUM_FEATURE_COLUMNS,
    *OPEN_INTEREST_FEATURE_COLUMNS,
]


@dataclass(frozen=True, slots=True)
class DerivativesFeatureConfig:
    funding_zscore_window_1h: int = 24 * 7
    funding_change_lookback_1h: int = 24
    premium_index_zscore_window_1h: int = 24 * 7
    premium_index_change_lookback_1h: int = 24
    open_interest_zscore_window_1h: int = 24 * 7
    open_interest_change_lookback_8h: int = 8
    open_interest_change_lookback_24h: int = 24

    @property
    def funding_zscore_window(self) -> int:
        return max(24, int(self.funding_zscore_window_1h))

    @property
    def funding_change_lookback(self) -> int:
        return max(1, int(self.funding_change_lookback_1h))

    @property
    def premium_zscore_window(self) -> int:
        return max(24, int(self.premium_index_zscore_window_1h))

    @property
    def premium_change_lookback(self) -> int:
        return max(1, int(self.premium_index_change_lookback_1h))

    @property
    def open_interest_zscore_window(self) -> int:
        return max(24, int(self.open_interest_zscore_window_1h))

    @property
    def open_interest_8h_lookback(self) -> int:
        return max(1, int(self.open_interest_change_lookback_8h))

    @property
    def open_interest_24h_lookback(self) -> int:
        return max(1, int(self.open_interest_change_lookback_24h))


def build_funding_frame(frame: pd.DataFrame, config: DerivativesFeatureConfig | None = None) -> pd.DataFrame:
    output = frame[["timestamp_ms"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_ms", *FUNDING_FEATURE_COLUMNS])
    return output.join(_build_funding_features(frame, config or DerivativesFeatureConfig()))[
        ["timestamp_ms", *FUNDING_FEATURE_COLUMNS]
    ]


def attach_funding_context(
    base_frame: pd.DataFrame,
    funding_frame: pd.DataFrame | None,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    return _attach_asof_context(
        base_frame,
        funding_frame,
        value_column="funding_rate",
        timestamp_column=timestamp_column,
    )


def attach_premium_index_context(
    base_frame: pd.DataFrame,
    premium_index_frame: pd.DataFrame | None,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    return _attach_asof_context(
        base_frame,
        premium_index_frame,
        value_column="premium_index_close",
        timestamp_column=timestamp_column,
    )


def attach_open_interest_context(
    base_frame: pd.DataFrame,
    open_interest_frame: pd.DataFrame | None,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    return _attach_asof_context(
        base_frame,
        open_interest_frame,
        value_column="open_interest",
        timestamp_column=timestamp_column,
    )


def build_premium_frame(frame: pd.DataFrame, config: DerivativesFeatureConfig | None = None) -> pd.DataFrame:
    output = frame[["timestamp_ms"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_ms", *PREMIUM_FEATURE_COLUMNS])
    return output.join(_build_premium_features(frame, config or DerivativesFeatureConfig()))[
        ["timestamp_ms", *PREMIUM_FEATURE_COLUMNS]
    ]


def build_open_interest_frame(frame: pd.DataFrame, config: DerivativesFeatureConfig | None = None) -> pd.DataFrame:
    output = frame[["timestamp_ms"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_ms", *OPEN_INTEREST_FEATURE_COLUMNS])
    return output.join(_build_open_interest_features(frame, config or DerivativesFeatureConfig()))[
        ["timestamp_ms", *OPEN_INTEREST_FEATURE_COLUMNS]
    ]


def _attach_asof_context(
    base_frame: pd.DataFrame,
    context_frame: pd.DataFrame | None,
    *,
    value_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    output = base_frame.copy().sort_values(timestamp_column).reset_index(drop=True)
    output[timestamp_column] = pd.to_datetime(output[timestamp_column], errors="coerce").astype("datetime64[ns]")
    if context_frame is None or context_frame.empty:
        output[value_column] = np.nan
        return output

    context = (
        context_frame[[timestamp_column, value_column]]
        .copy()
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )
    context[timestamp_column] = pd.to_datetime(context[timestamp_column], errors="coerce").astype("datetime64[ns]")
    return pd.merge_asof(
        output,
        context,
        on=timestamp_column,
        direction="backward",
    )


def _build_funding_features(frame: pd.DataFrame, config: DerivativesFeatureConfig) -> pd.DataFrame:
    if "funding_rate" not in frame.columns:
        raise ValueError("Funding features require raw column 'funding_rate'.")
    funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
    rolling_mean = funding_rate.rolling(config.funding_zscore_window).mean()
    rolling_std = funding_rate.rolling(config.funding_zscore_window).std().replace(0, np.nan)
    funding_zscore = (funding_rate - rolling_mean) / rolling_std
    return pd.DataFrame(
        {
            "funding_rate_8h": funding_rate,
            "funding_rate_zscore_7d": funding_zscore,
            "funding_rate_change_24h": funding_rate - funding_rate.shift(config.funding_change_lookback),
            "longs_overheated_1h": funding_zscore.clip(lower=0),
            "shorts_overheated_1h": (-funding_zscore).clip(lower=0),
        },
        index=frame.index,
    )


def _build_premium_features(frame: pd.DataFrame, config: DerivativesFeatureConfig) -> pd.DataFrame:
    if "premium_index_close" not in frame.columns:
        raise ValueError("Premium features require raw column 'premium_index_close'.")
    if "funding_rate" not in frame.columns:
        raise ValueError("Crowding premium features require raw column 'funding_rate'.")
    premium_index = pd.to_numeric(frame["premium_index_close"], errors="coerce")
    premium_mean = premium_index.rolling(config.premium_zscore_window).mean()
    premium_std = premium_index.rolling(config.premium_zscore_window).std().replace(0, np.nan)
    premium_zscore = (premium_index - premium_mean) / premium_std
    funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
    funding_mean = funding_rate.rolling(config.premium_zscore_window).mean()
    funding_std = funding_rate.rolling(config.premium_zscore_window).std().replace(0, np.nan)
    funding_zscore = (funding_rate - funding_mean) / funding_std
    return pd.DataFrame(
        {
            "premium_index_1h": premium_index,
            "premium_index_zscore_7d": premium_zscore,
            "premium_index_change_24h": premium_index - premium_index.shift(config.premium_change_lookback),
            "crowded_longs_score_1h": premium_zscore.clip(lower=0) * funding_zscore.clip(lower=0),
            "crowded_shorts_score_1h": (-premium_zscore).clip(lower=0) * (-funding_zscore).clip(lower=0),
        },
        index=frame.index,
    )


def _build_open_interest_features(frame: pd.DataFrame, config: DerivativesFeatureConfig) -> pd.DataFrame:
    if "open_interest" not in frame.columns:
        raise ValueError("Open interest features require raw column 'open_interest'.")
    if "funding_rate" not in frame.columns:
        raise ValueError("Feature 'open_interest_funding_crowding_1h' requires raw column 'funding_rate'.")
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce")
    rolling_mean = open_interest.rolling(config.open_interest_zscore_window).mean()
    rolling_std = open_interest.rolling(config.open_interest_zscore_window).std().replace(0, np.nan)
    open_interest_zscore = (open_interest - rolling_mean) / rolling_std
    oi_change_pct_8h = open_interest.pct_change(config.open_interest_8h_lookback)
    oi_change_pct_24h = open_interest.pct_change(config.open_interest_24h_lookback)
    funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
    funding_mean = funding_rate.rolling(config.open_interest_zscore_window).mean()
    funding_std = funding_rate.rolling(config.open_interest_zscore_window).std().replace(0, np.nan)
    funding_zscore = (funding_rate - funding_mean) / funding_std
    return pd.DataFrame(
        {
            "open_interest_zscore_7d": open_interest_zscore,
            "open_interest_change_pct_8h": oi_change_pct_8h,
            "open_interest_change_pct_24h": oi_change_pct_24h,
            "open_interest_funding_crowding_1h": oi_change_pct_24h * funding_zscore,
        },
        index=frame.index,
    )
