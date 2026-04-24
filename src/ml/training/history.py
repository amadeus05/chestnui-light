import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import config as cfg

from .diagnostics import build_fold_stability_payload
from .log import logger

def get_train_history_path(model_name):
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    return cfg.MODELS_DIR / f"{model_name}_train_history.json"


def load_train_history(history_path):
    if not history_path.exists():
        return []

    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Train history file is corrupted, resetting history: %s", history_path)
        return []

    if not isinstance(payload, list):
        logger.warning("Train history file has unexpected format, resetting history: %s", history_path)
        return []
    return payload


def extract_configured_signal_metrics(oos_metrics):
    threshold = round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5))), 2)
    threshold_key = f"{threshold:.2f}"
    threshold_metrics = (oos_metrics.get("probability_threshold_metrics") or {}).get(threshold_key, {})
    return {
        "configured_threshold": threshold,
        "signal_accuracy": threshold_metrics.get("signal_accuracy"),
        "signal_coverage": threshold_metrics.get("coverage"),
        "signal_rows": threshold_metrics.get("rows"),
    }


def build_train_history_entry(args, metrics, experiment_snapshot):
    oos_metrics = metrics["oos_metrics"]
    fold_stability = build_fold_stability_payload(metrics.get("fold_details", []))
    configured_signal_metrics = extract_configured_signal_metrics(oos_metrics)
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_name": args.model_name,
        "experiment": experiment_snapshot["experiment"],
        "labeling_profile": experiment_snapshot["labeling_profile"],
        "training_profile": experiment_snapshot["training_profile"],
        "accuracy": float(oos_metrics["accuracy"]),
        "balanced_accuracy": float(oos_metrics["balanced_accuracy"]),
        "f1_macro": float(oos_metrics["f1_macro"]),
        "roc_auc": float(oos_metrics["roc_auc"]),
        "pr_auc": float(oos_metrics["pr_auc"]),
        "mcc": float(oos_metrics["mcc"]),
        "fold_stability_pct": (
            float(fold_stability["accuracy_std"]) * 100 if fold_stability["accuracy_std"] is not None else None
        ),
        "fold_accuracy_range_pct": (
            float(fold_stability["accuracy_range"]) * 100 if fold_stability["accuracy_range"] is not None else None
        ),
        "fold_roc_auc_stability_pct": (
            float(fold_stability["roc_auc_std"]) * 100 if fold_stability["roc_auc_std"] is not None else None
        ),
        "configured_threshold": float(configured_signal_metrics["configured_threshold"]),
        "signal_accuracy": (
            float(configured_signal_metrics["signal_accuracy"])
            if configured_signal_metrics["signal_accuracy"] is not None else None
        ),
        "signal_coverage": (
            float(configured_signal_metrics["signal_coverage"])
            if configured_signal_metrics["signal_coverage"] is not None else None
        ),
        "signal_rows": (
            int(configured_signal_metrics["signal_rows"])
            if configured_signal_metrics["signal_rows"] is not None else None
        ),
        "median_best_iteration": int(metrics["median_best_iteration"]),
        "total_rows": int(metrics["total_rows"]),
        "feature_count": int(metrics["feature_count"]),
    }


def save_train_history(history_path, history_entry, limit=200):
    history = load_train_history(history_path)
    history.append(history_entry)
    history = history[-limit:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def format_compact_metric_value(value, percent=False, decimals=4):
    if value is None:
        return "-"
    if percent:
        return f"{float(value):.{decimals}f}%"
    return f"{float(value):.{decimals}f}"


def build_current_run_summary_lines(history_entry):
    configured_threshold = float(history_entry.get("configured_threshold", max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5)))))
    rows = [
        ("Accuracy", format_compact_metric_value(history_entry["accuracy"] * 100, percent=True, decimals=2)),
        (
            f"Signal acc @{configured_threshold:.2f}",
            format_compact_metric_value(
                (
                    float(history_entry["signal_accuracy"]) * 100
                    if history_entry.get("signal_accuracy") is not None else None
                ),
                percent=True,
                decimals=2,
            ),
        ),
        (
            f"Coverage @{configured_threshold:.2f}",
            format_compact_metric_value(
                (
                    float(history_entry["signal_coverage"]) * 100
                    if history_entry.get("signal_coverage") is not None else None
                ),
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", format_compact_metric_value(history_entry["mcc"], decimals=3)),
        ("ROC AUC", format_compact_metric_value(history_entry["roc_auc"], decimals=3)),
        ("PR AUC", format_compact_metric_value(history_entry["pr_auc"], decimals=3)),
        ("Fold stability", format_compact_metric_value(history_entry["fold_stability_pct"], percent=True, decimals=2)),
    ]
    metric_width = max(len("Metric"), *(len(name) for name, _ in rows))
    value_width = max(len("Current"), *(len(value) for _, value in rows))
    border = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    lines = [
        border,
        f"| {'Metric'.ljust(metric_width)} | {'Current'.ljust(value_width)} |",
        border,
    ]
    for name, value in rows:
        lines.append(f"| {name.ljust(metric_width)} | {value.ljust(value_width)} |")
    lines.append(border)
    return lines


def build_recent_runs_table_lines(history, limit=10):
    recent_entries = list(reversed(history[-limit:]))
    if not recent_entries:
        return ["No train history yet."]

    columns = [
        ("Run", lambda item: str(item.get("run_timestamp_utc", ""))[5:16]),
        ("Exp", lambda item: str(item.get("experiment", ""))[:18]),
        ("Acc", lambda item: format_compact_metric_value(item.get("accuracy", 0.0) * 100, percent=True, decimals=2)),
        (
            "Sig",
            lambda item: format_compact_metric_value(
                item.get("signal_accuracy") * 100 if item.get("signal_accuracy") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        (
            "Cov",
            lambda item: format_compact_metric_value(
                item.get("signal_coverage") * 100 if item.get("signal_coverage") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", lambda item: format_compact_metric_value(item.get("mcc"), decimals=3)),
        ("ROC", lambda item: format_compact_metric_value(item.get("roc_auc"), decimals=3)),
        ("PR", lambda item: format_compact_metric_value(item.get("pr_auc"), decimals=3)),
        ("Stab", lambda item: format_compact_metric_value(item.get("fold_stability_pct"), percent=True, decimals=2)),
        ("Rows", lambda item: str(item.get("total_rows", "-"))),
    ]

    rendered_rows = [[formatter(entry) for _, formatter in columns] for entry in recent_entries]
    widths = [
        max(len(header), *(len(row[idx]) for row in rendered_rows))
        for idx, (header, _) in enumerate(columns)
    ]

    def render_border():
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render_row(values):
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        render_border(),
        render_row([header for header, _ in columns]),
        render_border(),
    ]
    for row in rendered_rows:
        lines.append(render_row(row))
    lines.append(render_border())
    return lines


def log_train_history_summary(history_entry, history, limit=10):
    logger.info("=" * 72)
    logger.info("Current training summary:")
    for line in build_current_run_summary_lines(history_entry):
        logger.info(line)
    logger.info("Recent training runs (latest %s):", min(limit, len(history)))
    for line in build_recent_runs_table_lines(history, limit=limit):
        logger.info(line)
