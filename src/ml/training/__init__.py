"""LightGBM training pipeline (логика перенесена из train.py без изменений)."""
from signal_filter import resolve_event_filter_config

from .artifacts import (
    backup_existing_artifacts,
    build_feature_formulas_payload,
    log_feature_importance_ranking,
    save_directional_artifacts,
    top_feature_importance,
)
from .cli import parse_args
from .config_snapshot import build_experiment_snapshot, get_end_date_cutoff
from .constants import (
    CLASS_TO_LABEL,
    EXCLUDED_RAW_FEATURE_COLUMNS,
    LABEL_TO_CLASS,
    RESERVED_COLUMNS,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from .dataset import (
    build_symbol_row_profile,
    build_timestamp_profile,
    format_timestamp,
    load_training_frame,
)
from .diagnostics import (
    build_dataset_diagnostics,
    build_fold_boundary_summary,
    build_fold_stability_payload,
    build_period_payload,
)
from .features import (
    apply_feature_clip_bounds,
    build_feature_clip_bounds,
    get_clippable_feature_columns,
    resolve_internal_eval_plan,
    select_feature_columns,
)
from .history import (
    build_current_run_summary_lines,
    build_recent_runs_table_lines,
    build_train_history_entry,
    extract_configured_signal_metrics,
    format_compact_metric_value,
    get_train_history_path,
    load_train_history,
    log_train_history_summary,
    save_train_history,
)
from .importance import (
    aggregate_fold_importance,
    build_fold_importance_frame,
    build_fold_importance_summary,
)
from .metrics import evaluate_model
from .model_lgb import (
    build_model,
    compute_sample_weights,
    fit_model_with_internal_eval,
)
from .pipeline import main
from .production import train_production_model
from .splits import build_timestamp_splits, iter_monthly_timestamp_splits
from .walk_forward import walk_forward_validation

__all__ = [
    "CLASS_TO_LABEL",
    "EXCLUDED_RAW_FEATURE_COLUMNS",
    "LABEL_TO_CLASS",
    "RESERVED_COLUMNS",
    "SYMBOL_COLUMN",
    "TARGET_COLUMN",
    "TIMESTAMP_COLUMN",
    "aggregate_fold_importance",
    "apply_feature_clip_bounds",
    "backup_existing_artifacts",
    "build_current_run_summary_lines",
    "build_dataset_diagnostics",
    "build_experiment_snapshot",
    "build_feature_clip_bounds",
    "build_feature_formulas_payload",
    "build_fold_boundary_summary",
    "build_fold_importance_frame",
    "build_fold_importance_summary",
    "build_fold_stability_payload",
    "build_model",
    "build_period_payload",
    "build_recent_runs_table_lines",
    "build_symbol_row_profile",
    "build_timestamp_profile",
    "build_timestamp_splits",
    "build_train_history_entry",
    "compute_sample_weights",
    "evaluate_model",
    "extract_configured_signal_metrics",
    "fit_model_with_internal_eval",
    "format_compact_metric_value",
    "format_timestamp",
    "get_clippable_feature_columns",
    "get_end_date_cutoff",
    "get_train_history_path",
    "iter_monthly_timestamp_splits",
    "load_train_history",
    "load_training_frame",
    "log_feature_importance_ranking",
    "log_train_history_summary",
    "main",
    "parse_args",
    "resolve_event_filter_config",
    "resolve_internal_eval_plan",
    "save_directional_artifacts",
    "save_train_history",
    "select_feature_columns",
    "top_feature_importance",
    "train_production_model",
    "walk_forward_validation",
]
