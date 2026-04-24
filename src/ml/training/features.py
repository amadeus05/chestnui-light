import numpy as np
import pandas as pd
import config as cfg

from .constants import (
    EXCLUDED_RAW_FEATURE_COLUMNS,
    RESERVED_COLUMNS,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
)
from .log import logger

def select_feature_columns(dataset):
    feature_columns = []
    use_symbol_feature = bool(getattr(cfg, "USE_SYMBOL_FEATURE", True))
    disabled_feature_columns = set(getattr(cfg, "MANUAL_DISABLED_FEATURE_COLUMNS", []))

    for column in dataset.columns:
        if column in RESERVED_COLUMNS:
            continue
        if column in EXCLUDED_RAW_FEATURE_COLUMNS:
            continue
        if column in disabled_feature_columns:
            continue
        if column == SYMBOL_COLUMN:
            if use_symbol_feature:
                feature_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError("No usable feature columns found in the dataset.")
    if disabled_feature_columns:
        disabled_present = sorted(disabled_feature_columns.intersection(dataset.columns))
        if disabled_present:
            logger.info(
                "Config disabled %s feature columns, excluding them from training: %s",
                len(disabled_present),
                ", ".join(disabled_present),
            )
    return feature_columns


def get_clippable_feature_columns(dataset, feature_columns):
    clippable_columns = []
    for column in feature_columns:
        if column == SYMBOL_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            clippable_columns.append(column)
    return clippable_columns


def build_feature_clip_bounds(train_df, feature_columns):
    """Compute winsorization bounds from the training set quantiles."""
    if not bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)):
        return {}

    lower_q = float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01))
    upper_q = float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99))
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("FEATURE_CLIP_LOWER_Q and FEATURE_CLIP_UPPER_Q must satisfy 0 <= lower < upper <= 1.")

    clip_bounds = {}
    for column in get_clippable_feature_columns(train_df, feature_columns):
        series = train_df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        if pd.isna(lower) or pd.isna(upper):
            continue
        clip_bounds[column] = {"lower": float(lower), "upper": float(upper)}
    return clip_bounds


def apply_feature_clip_bounds(frame, clip_bounds):
    """Apply precomputed winsorization bounds to a dataframe."""
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def resolve_internal_eval_plan(y_train: pd.Series) -> dict:
    """
    Pick a suffix eval slice that remains time-ordered but is less class-skewed.
    """
    n_rows = int(len(y_train))
    if n_rows <= 1:
        base_rate = float(y_train.mean()) if n_rows else 0.5
        return {
            "eval_size": 1,
            "eval_fraction": 1.0,
            "fit_rate": base_rate,
            "eval_rate": base_rate,
            "was_expanded": False,
        }

    min_fraction = float(getattr(cfg, "INTERNAL_EVAL_MIN_FRACTION", 0.15))
    max_fraction = float(getattr(cfg, "INTERNAL_EVAL_MAX_FRACTION", 0.40))
    step_fraction = float(getattr(cfg, "INTERNAL_EVAL_STEP_FRACTION", 0.05))
    max_rate_diff = float(getattr(cfg, "INTERNAL_EVAL_MAX_CLASS_RATE_DIFF", 0.08))

    min_fraction = min(max(min_fraction, 0.05), 0.45)
    max_fraction = min(max(max_fraction, min_fraction), 0.50)
    step_fraction = min(max(step_fraction, 0.01), 0.10)

    candidate_fractions = []
    current_fraction = min_fraction
    while current_fraction <= max_fraction + 1e-9:
        candidate_fractions.append(round(current_fraction, 4))
        current_fraction += step_fraction

    best_plan = None
    for idx, fraction in enumerate(candidate_fractions):
        eval_size = max(1, int(n_rows * fraction))
        if eval_size >= n_rows:
            eval_size = n_rows - 1
        if eval_size <= 0:
            continue

        y_fit = y_train.iloc[:-eval_size]
        y_eval = y_train.iloc[-eval_size:]
        if y_fit.empty:
            continue
        if len(set(y_fit.unique().tolist())) < 2 or len(set(y_eval.unique().tolist())) < 2:
            continue

        fit_rate = float(y_fit.mean())
        eval_rate = float(y_eval.mean())
        plan = {
            "eval_size": int(eval_size),
            "eval_fraction": float(eval_size / n_rows),
            "fit_rate": fit_rate,
            "eval_rate": eval_rate,
            "rate_diff": abs(eval_rate - fit_rate),
            "was_expanded": idx > 0,
        }
        if best_plan is None or plan["rate_diff"] < best_plan["rate_diff"]:
            best_plan = plan
        if plan["rate_diff"] <= max_rate_diff:
            return plan

    if best_plan is not None:
        return best_plan

    eval_size = max(1, int(n_rows * min_fraction))
    if eval_size >= n_rows:
        eval_size = n_rows - 1
    y_fit = y_train.iloc[:-eval_size]
    y_eval = y_train.iloc[-eval_size:]
    return {
        "eval_size": int(eval_size),
        "eval_fraction": float(eval_size / n_rows),
        "fit_rate": float(y_fit.mean()) if not y_fit.empty else float(y_train.mean()),
        "eval_rate": float(y_eval.mean()) if not y_eval.empty else float(y_train.mean()),
        "was_expanded": False,
    }
