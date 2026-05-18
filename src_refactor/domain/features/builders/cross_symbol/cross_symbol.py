from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MARKET_CONTEXT_COLUMNS = ["market_breadth_ema_fast_slow_1h"]
BASE_CROSS_SECTIONAL_COLUMNS = ["cross_sectional_rank_ema_fast_slow_1h"]
HTF_CROSS_SECTIONAL_COLUMNS = ["cross_sectional_rank_4h"]
HTF_MARKET_CONTEXT_COLUMNS = [
    "market_breadth_pos_return_4h_3",
    "market_dispersion_return_4h_3",
    "market_breadth_pos_return_4h_14",
    "market_mean_return_4h_14",
    "market_avg_ema_slope_4h",
    "market_abs_avg_ema_slope_4h",
]


def enrich_base_market_context(
    feature_map: dict[str, pd.DataFrame],
    timestamp_column: str = "timestamp_ms",
) -> dict[str, pd.DataFrame]:
    context_df = _build_market_breadth_context(feature_map, timestamp_column)
    return {
        symbol: _merge_context(frame, context_df, BASE_MARKET_CONTEXT_COLUMNS, timestamp_column)
        for symbol, frame in feature_map.items()
    }


def enrich_base_cross_sectional_rank(
    feature_map: dict[str, pd.DataFrame],
    timestamp_column: str = "timestamp_ms",
) -> dict[str, pd.DataFrame]:
    rank_map = _build_rank_map(feature_map, timestamp_column)
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in feature_map.items():
        base = frame.copy()
        symbol_rank = rank_map.get(symbol)
        if symbol_rank is None or symbol_rank.empty:
            base["cross_sectional_rank_ema_fast_slow_1h"] = pd.NA
        else:
            base = base.merge(symbol_rank, on=timestamp_column, how="left")
        output[symbol] = base
    return output


def enrich_htf_market_context(
    htf_feature_map: dict[str, pd.DataFrame],
    timestamp_column: str = "timestamp_ms",
) -> dict[str, pd.DataFrame]:
    context_df = _build_htf_market_context(htf_feature_map, timestamp_column)
    return {
        symbol: _merge_context(frame, context_df, HTF_MARKET_CONTEXT_COLUMNS, timestamp_column)
        for symbol, frame in htf_feature_map.items()
    }


def enrich_htf_cross_sectional_rank(
    htf_feature_map: dict[str, pd.DataFrame],
    timestamp_column: str = "timestamp_ms",
) -> dict[str, pd.DataFrame]:
    rank_map = _build_htf_rank_map(htf_feature_map, timestamp_column)
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in htf_feature_map.items():
        base = frame.copy()
        symbol_rank = rank_map.get(symbol)
        if symbol_rank is None or symbol_rank.empty:
            base["cross_sectional_rank_4h"] = pd.NA
        else:
            base = base.merge(symbol_rank, on=timestamp_column, how="left")
        output[symbol] = base
    return output


def _build_market_breadth_context(feature_map: dict[str, pd.DataFrame], timestamp_column: str) -> pd.DataFrame:
    frames = []
    for source_df in feature_map.values():
        if "ema_fast_slow" not in source_df.columns:
            continue
        frame = source_df[[timestamp_column, "ema_fast_slow"]].dropna(subset=["ema_fast_slow"]).copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[timestamp_column, *BASE_MARKET_CONTEXT_COLUMNS])
    context_source = pd.concat(frames, ignore_index=True)
    grouped = context_source.groupby(timestamp_column)["ema_fast_slow"]
    context_df = pd.DataFrame({timestamp_column: grouped.size().index})
    context_df["market_breadth_ema_fast_slow_1h"] = grouped.apply(lambda series: float((series > 0).mean())).values
    return context_df


def _build_rank_map(feature_map: dict[str, pd.DataFrame], timestamp_column: str) -> dict[str, pd.DataFrame]:
    rank_frames = []
    for symbol, source_df in feature_map.items():
        if "ema_fast_slow" not in source_df.columns:
            continue
        frame = source_df[[timestamp_column, "ema_fast_slow"]].dropna(subset=["ema_fast_slow"]).copy()
        if frame.empty:
            continue
        frame["symbol"] = symbol
        rank_frames.append(frame)
    if not rank_frames:
        return {}
    rank_df = pd.concat(rank_frames, ignore_index=True)
    rank_df["cross_sectional_rank_ema_fast_slow_1h"] = rank_df.groupby(timestamp_column)["ema_fast_slow"].transform(
        _normalize_rank
    )
    return {
        symbol: rank_df.loc[
            rank_df["symbol"] == symbol,
            [timestamp_column, "cross_sectional_rank_ema_fast_slow_1h"],
        ].copy()
        for symbol in feature_map
    }


def _build_htf_market_context(htf_feature_map: dict[str, pd.DataFrame], timestamp_column: str) -> pd.DataFrame:
    frames = []
    for source_df in htf_feature_map.values():
        available_columns = {timestamp_column}
        for column in ("return_4h_3", "return_4h_14", "ema_slope_4h"):
            if column in source_df.columns:
                available_columns.add(column)
        if len(available_columns) == 1:
            continue
        frame = source_df[list(available_columns)].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[timestamp_column, *HTF_MARKET_CONTEXT_COLUMNS])

    context_source = pd.concat(frames, ignore_index=True)
    context_df = pd.DataFrame(
        {timestamp_column: pd.Index(sorted(context_source[timestamp_column].dropna().unique()), name=timestamp_column)}
    )

    if "return_4h_3" in context_source.columns:
        grouped_return_4h_3 = context_source.dropna(subset=["return_4h_3"]).groupby(timestamp_column)["return_4h_3"]
        if grouped_return_4h_3.ngroups != 0:
            context_df = context_df.merge(
                grouped_return_4h_3.apply(lambda series: float((series > 0).mean()))
                .rename("market_breadth_pos_return_4h_3")
                .reset_index(),
                on=timestamp_column,
                how="left",
            )
            context_df = context_df.merge(
                grouped_return_4h_3.apply(lambda series: float(np.nanstd(series.to_numpy(dtype=float), ddof=0)))
                .rename("market_dispersion_return_4h_3")
                .reset_index(),
                on=timestamp_column,
                how="left",
            )

    if "return_4h_14" in context_source.columns:
        grouped_return_4h_14 = context_source.dropna(subset=["return_4h_14"]).groupby(timestamp_column)[
            "return_4h_14"
        ]
        if grouped_return_4h_14.ngroups != 0:
            context_df = context_df.merge(
                grouped_return_4h_14.apply(lambda series: float((series > 0).mean()))
                .rename("market_breadth_pos_return_4h_14")
                .reset_index(),
                on=timestamp_column,
                how="left",
            )
            context_df = context_df.merge(
                grouped_return_4h_14.mean().astype(float).rename("market_mean_return_4h_14").reset_index(),
                on=timestamp_column,
                how="left",
            )

    if "ema_slope_4h" in context_source.columns:
        grouped_ema_slope_4h = context_source.dropna(subset=["ema_slope_4h"]).groupby(timestamp_column)["ema_slope_4h"]
        if grouped_ema_slope_4h.ngroups != 0:
            context_df = context_df.merge(
                grouped_ema_slope_4h.mean().astype(float).rename("market_avg_ema_slope_4h").reset_index(),
                on=timestamp_column,
                how="left",
            )
            context_df = context_df.merge(
                grouped_ema_slope_4h.apply(lambda series: float(np.nanmean(np.abs(series.to_numpy(dtype=float)))))
                .rename("market_abs_avg_ema_slope_4h")
                .reset_index(),
                on=timestamp_column,
                how="left",
            )
    return context_df


def _build_htf_rank_map(htf_feature_map: dict[str, pd.DataFrame], timestamp_column: str) -> dict[str, pd.DataFrame]:
    rank_frames = []
    for symbol, source_df in htf_feature_map.items():
        if "return_4h_3" not in source_df.columns:
            continue
        frame = source_df[[timestamp_column, "return_4h_3"]].dropna(subset=["return_4h_3"]).copy()
        if frame.empty:
            continue
        frame["symbol"] = symbol
        rank_frames.append(frame)
    if not rank_frames:
        return {}
    rank_df = pd.concat(rank_frames, ignore_index=True)
    rank_df["cross_sectional_rank_4h"] = rank_df.groupby(timestamp_column)["return_4h_3"].transform(_normalize_rank)
    return {
        symbol: rank_df.loc[rank_df["symbol"] == symbol, [timestamp_column, "cross_sectional_rank_4h"]].copy()
        for symbol in htf_feature_map
    }


def _merge_context(
    frame: pd.DataFrame,
    context_df: pd.DataFrame,
    context_columns: list[str],
    timestamp_column: str,
) -> pd.DataFrame:
    output = frame.copy()
    if context_df.empty:
        for column in context_columns:
            output[column] = pd.NA
        return output
    return output.merge(context_df, on=timestamp_column, how="left")


def _normalize_rank(series: pd.Series) -> pd.Series:
    if len(series) == 1:
        return pd.Series(0.5, index=series.index)
    ranked = series.rank(method="average")
    return (ranked - 1) / (len(series) - 1)
