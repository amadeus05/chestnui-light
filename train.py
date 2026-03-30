import argparse
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

import config as cfg
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "Target"
TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
RESERVED_COLUMNS = {
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    "barrier_stop_pct",
    "barrier_take_pct",
}
EXCLUDED_RAW_FEATURE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}
TEST_FEATURE_COLUMNS = {
    "range_compression_1h",
    "distance_to_session_high_1h",
    "distance_to_session_low_1h",
    "trend_persistence_score_12",
    "trend_persistence_score_24",
    "trend_efficiency_24h",
    "slope_acceleration_1h_12_24",
    "ema_slope_acceleration_1h",
    "volatility_acceleration_1h",
    "hour_sin_1h",
    "hour_cos_1h",
    "is_weekend_1h",
    "relative_strength_vs_btc_24h",
    "beta_to_btc_24h",
    "residual_return_24h",
    "cross_sectional_rank_ema_fast_slow_1h",
    "market_breadth_ema_fast_slow_1h",
    "market_breadth_pos_return_4h_3",
    "market_dispersion_return_4h_3",
    "delta_market_breadth_ema_fast_slow_1h",
    "market_breadth_ema_fast_slow_1h_zscore",
    "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
    "trend_efficiency_24h_x_volatility_regime_change_1h",
}
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {0: -1, 1: 1}


def get_end_date_cutoff():
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return None
    return pd.to_datetime(end_date, errors="coerce")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM classifier on ETL feature tables.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to load, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation share for chronological split.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Holdout test share for chronological split.")
    parser.add_argument("--model-name", default="lightgbm_target", help="Base filename for saved artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--prod-train",
        action="store_true",
        default=bool(getattr(cfg, "ENABLE_PROD_TRAINING", False)),
        help="After validation, retrain the final model on the full dataset.",
    )
    return parser.parse_args()


def load_training_frame(db_path, symbols):
    repository = HistoricalKlineRepository(db_path=db_path)
    dataset = repository.load_feature_dataset(symbols)
    dataset = dataset.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is not None and not pd.isna(end_cutoff):
        before_rows = len(dataset)
        dataset = dataset.loc[dataset[TIMESTAMP_COLUMN] <= end_cutoff].copy()
        logger.info(
            "Applied END_DATE cutoff at %s: kept %s/%s rows",
            end_cutoff,
            len(dataset),
            before_rows,
        )

    raw_labels = dataset[TARGET_COLUMN].astype(int)
    unknown_labels = sorted(set(raw_labels.unique()) - {-1, 0, 1})
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {TARGET_COLUMN}: {unknown_labels}")

    event_filter_config = resolve_event_filter_config()
    candidate_mask = build_candidate_event_mask(dataset, event_filter_config)
    dataset.attrs["event_filter_config"] = event_filter_config
    dataset.attrs["candidate_rows"] = int(candidate_mask.sum())
    dataset.attrs["excluded_by_event_filter_rows"] = int((~candidate_mask).sum())
    dataset = dataset.loc[candidate_mask].copy()

    directional_mask = dataset[TARGET_COLUMN].astype(int) != 0
    excluded_non_directional_rows = int((~directional_mask).sum())
    dataset = dataset.loc[directional_mask].copy()
    raw_directional_labels = dataset[TARGET_COLUMN].astype(int)
    dataset[TARGET_COLUMN] = raw_directional_labels.map({-1: 0, 1: 1})
    dataset[SYMBOL_COLUMN] = dataset[SYMBOL_COLUMN].astype("category")
    dataset.attrs["excluded_non_directional_rows"] = excluded_non_directional_rows
    return dataset


def select_feature_columns(dataset):
    feature_columns = []
    use_symbol_feature = bool(getattr(cfg, "USE_SYMBOL_FEATURE", True))
    disabled_feature_columns = set(getattr(cfg, "MANUAL_DISABLED_FEATURE_COLUMNS", []))
    if not bool(getattr(cfg, "ENABLE_TEST_FEATURES", False)):
        disabled_feature_columns.update(TEST_FEATURE_COLUMNS)

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
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def build_period_payload(frame):
    if frame.empty:
        return None
    return {
        "start": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "end": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


def time_split(dataset, val_size, test_size):
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if (val_size + test_size) >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")

    unique_timestamps = dataset[TIMESTAMP_COLUMN].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_timestamps) < 3:
        raise RuntimeError("Need at least 3 unique timestamps for a chronological train/validation/test split.")

    train_end_idx = int(len(unique_timestamps) * (1 - val_size - test_size))
    valid_end_idx = int(len(unique_timestamps) * (1 - test_size))

    train_end_idx = max(1, train_end_idx)
    valid_end_idx = max(train_end_idx + 1, valid_end_idx)
    valid_end_idx = min(valid_end_idx, len(unique_timestamps) - 1)
    if train_end_idx >= valid_end_idx:
        raise RuntimeError("Chronological split is too small for separate validation and test windows.")

    valid_start_ts = unique_timestamps.iloc[train_end_idx]
    test_start_ts = unique_timestamps.iloc[valid_end_idx]

    train_df = dataset.loc[dataset[TIMESTAMP_COLUMN] < valid_start_ts].copy()
    valid_df = dataset.loc[
        (dataset[TIMESTAMP_COLUMN] >= valid_start_ts) & (dataset[TIMESTAMP_COLUMN] < test_start_ts)
    ].copy()
    test_df = dataset.loc[dataset[TIMESTAMP_COLUMN] >= test_start_ts].copy()
    return train_df, valid_df, test_df


def validate_split(train_df, valid_df, test_df):
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(
            "Train/validation/test split produced an empty part. Adjust split sizes or prepare more data."
        )

    train_last_ts = train_df[TIMESTAMP_COLUMN].max()
    valid_first_ts = valid_df[TIMESTAMP_COLUMN].min()
    valid_last_ts = valid_df[TIMESTAMP_COLUMN].max()
    test_first_ts = test_df[TIMESTAMP_COLUMN].min()
    if train_last_ts >= valid_first_ts:
        raise RuntimeError("Train/validation split has overlapping timestamps, which would leak validation context.")
    if valid_last_ts >= test_first_ts:
        raise RuntimeError("Validation/test split has overlapping timestamps, which would leak holdout context.")

    train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())
    if len(train_classes) < 2:
        human_labels = [CLASS_TO_LABEL[class_id] for class_id in train_classes]
        raise RuntimeError(f"Training split has too few classes for LightGBM: {human_labels}")


def build_model(seed, n_estimators=800):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_validation_model(train_df, valid_df, feature_columns, seed):
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]

    model = build_model(seed=seed)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def evaluate_model(model, eval_df, feature_columns, split_name):
    x_eval = eval_df[feature_columns]
    y_true = eval_df[TARGET_COLUMN]
    y_pred = model.predict(x_eval)
    y_proba = model.predict_proba(x_eval)
    p_long = y_proba[:, 1]

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["short", "long"],
        output_dict=True,
        zero_division=0,
    )

    confidence_thresholds = sorted(
        {
            round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5))), 2),
            0.55,
            0.60,
            0.65,
            0.70,
        }
    )
    probability_threshold_metrics = {}
    p_short = y_proba[:, 0]
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    for threshold in confidence_thresholds:
        threshold = float(threshold)
        signal = np.full(len(eval_df), -1, dtype=int)
        signal[p_long >= threshold] = 1
        signal[p_short >= threshold] = 0
        mask = signal != -1
        selected = int(mask.sum())
        coverage = float(selected / len(eval_df)) if len(eval_df) else 0.0
        long_signals = int((signal == 1).sum())
        short_signals = int((signal == 0).sum())
        no_trade = int((signal == -1).sum())

        if selected == 0:
            probability_threshold_metrics[f"{threshold:.2f}"] = {
                "rows": 0,
                "coverage": coverage,
                "long_signals": long_signals,
                "short_signals": short_signals,
                "no_trade": no_trade,
                "signal_accuracy": None,
                "signal_balanced_accuracy": None,
                "signal_f1_macro": None,
                "signal_confusion_matrix": None,
                "signal_classification_report": None,
                "long_precision": None,
                "short_precision": None,
                "long_recall_all": 0.0,
                "short_recall_all": 0.0,
            }
            continue

        subset_y_true = y_true_series.loc[mask]
        subset_y_pred = pd.Series(signal[mask], index=subset_y_true.index)
        subset_report = classification_report(
            subset_y_true,
            subset_y_pred,
            labels=[0, 1],
            target_names=["short", "long"],
            output_dict=True,
            zero_division=0,
        )
        long_tp = int(((signal == 1) & (y_true_series.values == 1)).sum())
        short_tp = int(((signal == 0) & (y_true_series.values == 0)).sum())
        total_true_long = int((y_true_series.values == 1).sum())
        total_true_short = int((y_true_series.values == 0).sum())

        probability_threshold_metrics[f"{threshold:.2f}"] = {
            "rows": selected,
            "coverage": coverage,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "no_trade": no_trade,
            "signal_accuracy": float(accuracy_score(subset_y_true, subset_y_pred)),
            "signal_balanced_accuracy": float(balanced_accuracy_score(subset_y_true, subset_y_pred)),
            "signal_f1_macro": float(f1_score(subset_y_true, subset_y_pred, average="macro")),
            "signal_confusion_matrix": confusion_matrix(subset_y_true, subset_y_pred, labels=[0, 1]).tolist(),
            "signal_classification_report": subset_report,
            "long_precision": float(long_tp / long_signals) if long_signals > 0 else None,
            "short_precision": float(short_tp / short_signals) if short_signals > 0 else None,
            "long_recall_all": float(long_tp / total_true_long) if total_true_long > 0 else 0.0,
            "short_recall_all": float(short_tp / total_true_short) if total_true_short > 0 else 0.0,
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, p_long)),
        "pr_auc": float(average_precision_score(y_true, p_long)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
        f"{split_name}_rows": int(len(eval_df)),
        "probability_threshold_metrics": probability_threshold_metrics,
    }
    return metrics


def retrain_full_model(dataset, feature_columns, seed, best_iteration):
    n_estimators = int(best_iteration) if best_iteration and best_iteration > 0 else 200
    final_model = build_model(seed=seed, n_estimators=n_estimators)
    final_model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
    )
    return final_model


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


def save_directional_artifacts(model, metrics, dataset, feature_columns, clip_bounds, args):
    cfg.MODELS_DIR.mkdir(exist_ok=True)

    model_path = cfg.MODELS_DIR / f"{args.model_name}.joblib"
    metrics_path = cfg.MODELS_DIR / f"{args.model_name}_metrics.json"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"
    importance_path = cfg.MODELS_DIR / f"{args.model_name}_feature_importance.csv"

    payload = {
        "feature_columns": feature_columns,
        "label_mapping": {"short": 0, "long": 1},
        "inverse_label_mapping": {str(key): value for key, value in CLASS_TO_LABEL.items()},
        "symbols": list(args.symbols),
        "rows": int(len(dataset)),
        "prod_train": bool(args.prod_train),
        "task_type": "binary_directional",
        "train_period": metrics.get("train_period"),
        "validation_period": metrics.get("validation_period"),
        "test_period": metrics.get("test_period"),
        "split_sizes": metrics.get("split_sizes"),
        "event_filter": metrics.get("event_filter"),
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": clip_bounds,
        },
    }

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    top_feature_importance(model, feature_columns).to_csv(importance_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved feature metadata to %s", features_path)
    logger.info("Saved feature importance to %s", importance_path)


def main():
    try:
        args = parse_args()
        dataset = load_training_frame(args.db_path, args.symbols)
        feature_columns = select_feature_columns(dataset)

        logger.info("Loaded %s rows with %s features", len(dataset), len(feature_columns))
        logger.info("Using symbols: %s", ", ".join(args.symbols))
        logger.info(
            "Candidate universe: kept %s rows after deterministic event filter, excluded %s rows",
            int(dataset.attrs.get("candidate_rows", len(dataset))),
            int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
        )
        logger.info(
            "Directional baseline inside candidate universe: excluded %s non-directional rows with Target=0 before split",
            int(dataset.attrs.get("excluded_non_directional_rows", 0)),
        )

        train_df, valid_df, test_df = time_split(dataset, args.val_size, args.test_size)
        validate_split(train_df, valid_df, test_df)
        clip_bounds = build_feature_clip_bounds(train_df, feature_columns)
        train_df = apply_feature_clip_bounds(train_df, clip_bounds)
        valid_df = apply_feature_clip_bounds(valid_df, clip_bounds)
        test_df = apply_feature_clip_bounds(test_df, clip_bounds)
        logger.info(
            "Chronological split: train=%s rows, valid=%s rows, test=%s rows | valid starts at %s | test starts at %s",
            len(train_df),
            len(valid_df),
            len(test_df),
            valid_df[TIMESTAMP_COLUMN].iloc[0],
            test_df[TIMESTAMP_COLUMN].iloc[0],
        )
        if clip_bounds:
            logger.info(
                "Feature clipping enabled: %s numeric columns clipped to [%.2f%%, %.2f%%] train percentiles",
                len(clip_bounds),
                float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)) * 100,
                float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)) * 100,
            )

        model = train_validation_model(train_df, valid_df, feature_columns, args.seed)
        validation_metrics = evaluate_model(model, valid_df, feature_columns, split_name="validation")
        test_metrics = evaluate_model(model, test_df, feature_columns, split_name="test")
        metrics = {
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_iteration": int(model.best_iteration_ or model.n_estimators_),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_columns)),
            "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            "candidate_rows": int(dataset.attrs.get("candidate_rows", len(dataset))),
            "excluded_by_event_filter_rows": int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
            "event_filter": dataset.attrs.get("event_filter_config"),
            "train_period": build_period_payload(train_df),
            "validation_period": build_period_payload(valid_df),
            "test_period": build_period_payload(test_df),
            "split_sizes": {
                "validation": float(args.val_size),
                "test": float(args.test_size),
            },
        }

        logger.info(
            "Validation metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
            validation_metrics["accuracy"],
            validation_metrics["balanced_accuracy"],
            validation_metrics["f1_macro"],
            validation_metrics["roc_auc"],
            validation_metrics["pr_auc"],
            validation_metrics["mcc"],
        )
        logger.info(
            "Holdout test metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
            test_metrics["accuracy"],
            test_metrics["balanced_accuracy"],
            test_metrics["f1_macro"],
            test_metrics["roc_auc"],
            test_metrics["pr_auc"],
            test_metrics["mcc"],
        )

        model_to_save = model
        clip_bounds_to_save = clip_bounds
        if args.prod_train:
            logger.warning(
                "ENABLE_PROD_TRAINING is enabled: the saved model will be retrained on the full dataset, "
                "including the holdout test window. Use the saved test metrics for evaluation, but do not treat "
                "subsequent backtests with this retrained artifact as out-of-sample."
            )
            clip_bounds_to_save = build_feature_clip_bounds(dataset, feature_columns)
            dataset = apply_feature_clip_bounds(dataset, clip_bounds_to_save)
            model_to_save = retrain_full_model(dataset, feature_columns, args.seed, metrics["best_iteration"])

        log_feature_importance_ranking(model_to_save, feature_columns)
        save_directional_artifacts(model_to_save, metrics, dataset, feature_columns, clip_bounds_to_save, args)
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
