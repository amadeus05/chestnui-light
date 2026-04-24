import json
import shutil
import hashlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import config as cfg

from src.features import MasterFeatureBuilder
from src.features.models.feature_spec import serialize_feature_specs

from .constants import CLASS_TO_LABEL, SYMBOL_COLUMN, TIMESTAMP_COLUMN
from .diagnostics import build_period_payload
from .log import logger


def top_feature_importance(model, feature_columns, limit=25):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    return importance.head(limit)


def log_feature_importance_ranking(model, feature_columns):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    logger.info("Feature importance ranking:")
    for rank, row in enumerate(importance.itertuples(index=False), start=1):
        logger.info(
            "%s. %s | gain=%.6f | split=%s",
            rank,
            row.feature,
            float(row.importance_gain),
            int(row.importance_split),
        )


def build_feature_formulas_payload(feature_columns, model_name, symbols, experiment_snapshot):
    builder = MasterFeatureBuilder()
    allowed_untracked = {SYMBOL_COLUMN}
    tracked_feature_columns = [column for column in feature_columns if column not in allowed_untracked]
    feature_specs = builder.collect_feature_specs(set(tracked_feature_columns))
    missing_specs = sorted(set(tracked_feature_columns) - set(feature_specs))
    if missing_specs:
        raise RuntimeError(
            "Missing FeatureSpec metadata for trained features: " + ", ".join(missing_specs)
        )

    return {
        "model_name": model_name,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "feature_columns": list(feature_columns),
        "tracked_feature_columns": tracked_feature_columns,
        "untracked_feature_columns": [column for column in feature_columns if column in allowed_untracked],
        "symbols": list(symbols),
        "experiment": experiment_snapshot,
        "features": serialize_feature_specs(feature_specs, cfg),
    }


def build_payload_fingerprint(payload):
    stable_payload = dict(payload)
    stable_payload.pop("generated_at_utc", None)
    canonical = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_directional_artifacts(
    model,
    metrics,
    dataset,
    feature_columns,
    clip_bounds,
    fold_importance,
    args,
    experiment_snapshot,
):
    cfg.MODELS_DIR.mkdir(exist_ok=True)

    model_path = cfg.MODELS_DIR / f"{args.model_name}.joblib"
    metrics_path = cfg.MODELS_DIR / f"{args.model_name}_metrics.json"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"
    importance_path = cfg.MODELS_DIR / f"{args.model_name}_feature_importance.csv"
    fold_importance_path = cfg.MODELS_DIR / f"{args.model_name}_fold_feature_importance.csv"
    feature_formulas_path = cfg.MODELS_DIR / f"{args.model_name}_feature_formulas.json"
    artifact_paths = [model_path, metrics_path, features_path, importance_path, fold_importance_path, feature_formulas_path]

    feature_formulas_payload = build_feature_formulas_payload(
        feature_columns=feature_columns,
        model_name=args.model_name,
        symbols=args.symbols,
        experiment_snapshot=experiment_snapshot,
    )
    feature_formulas_hash = build_payload_fingerprint(feature_formulas_payload)

    payload = {
        "feature_columns": feature_columns,
        "label_mapping": {"short": 0, "long": 1},
        "inverse_label_mapping": {str(key): value for key, value in CLASS_TO_LABEL.items()},
        "symbols": list(args.symbols),
        "rows": int(len(dataset)),
        "task_type": "binary_directional",
        "train_period": build_period_payload(dataset),
        "wfv_n_splits": args.n_splits,
        "wfv_purge_gap": args.purge_gap,
        "wfv_split_mode": args.split_mode,
        "wfv_monthly_train_months": args.monthly_train_months,
        "wfv_monthly_test_months": args.monthly_test_months,
        "wfv_monthly_window_mode": args.monthly_window_mode,
        "event_filter": metrics.get("event_filter"),
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": clip_bounds,
        },
        "feature_formulas_artifact": feature_formulas_path.name,
        "feature_formulas_sha256": feature_formulas_hash,
    }

    backup_existing_artifacts(artifact_paths, args.model_name)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    feature_formulas_path.write_text(json.dumps(feature_formulas_payload, indent=2), encoding="utf-8")
    top_feature_importance(model, feature_columns).to_csv(importance_path, index=False)
    if fold_importance is not None and not fold_importance.empty:
        fold_importance.to_csv(fold_importance_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved feature metadata to %s", features_path)
    logger.info("Saved feature formulas to %s", feature_formulas_path)
    logger.info("Saved feature importance to %s", importance_path)
    if fold_importance is not None and not fold_importance.empty:
        logger.info("Saved fold feature importance to %s", fold_importance_path)


def backup_existing_artifacts(paths, model_name):
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = cfg.MODELS_DIR / "backups" / f"{model_name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing_paths:
        shutil.copy2(path, backup_dir / path.name)

    logger.info(
        "Backed up %s existing model artifacts to %s",
        len(existing_paths),
        backup_dir,
    )
