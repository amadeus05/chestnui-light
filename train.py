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
from sklearn.model_selection import TimeSeriesSplit

import config as cfg
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & label mappings
# ---------------------------------------------------------------------------
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
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {0: -1, 1: 1}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI / data loading
# ═══════════════════════════════════════════════════════════════════════════

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
    parser.add_argument("--model-name", default="lightgbm_target", help="Base filename for saved artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of expanding-window folds for Walk-Forward Validation.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=12,
        help="Purge gap in timestamps between train and test folds to avoid target leakage.",
    )
    return parser.parse_args()


def load_training_frame(db_path, symbols):
    """Load dataset, filter events, keep only directional labels {-1, 1} → {0, 1}."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  Feature selection & clipping
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Model building
# ═══════════════════════════════════════════════════════════════════════════

def build_model(seed, n_estimators=800):
    """Instantiate LightGBM binary classifier. class_weight=None → calibrated probs."""
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
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation — accepts raw vectors, not a dataframe
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, y_proba, split_name, n_rows=None):
    """
    Compute a full metrics dictionary from pre-assembled OOS vectors.

    Parameters
    ----------
    y_true   : array-like of {0, 1}
    y_pred   : array-like of {0, 1}
    y_proba  : ndarray of shape (N, 2) — class probabilities
    split_name : str, used as a label in the metrics dict
    n_rows   : optional int, total rows evaluated (defaults to len(y_true))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    p_long = y_proba[:, 1]
    if n_rows is None:
        n_rows = len(y_true)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["short", "long"],
        output_dict=True,
        zero_division=0,
    )

    # ---- Confidence-threshold breakdown ----
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
        signal = np.full(n_rows, -1, dtype=int)
        signal[p_long >= threshold] = 1
        signal[p_short >= threshold] = 0
        mask = signal != -1
        selected = int(mask.sum())
        coverage = float(selected / n_rows) if n_rows else 0.0
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
        f"{split_name}_rows": n_rows,
        "probability_threshold_metrics": probability_threshold_metrics,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Walk-Forward Validation  (Expanding Window + Purge Gap)
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_validation(dataset, feature_columns, seed, n_splits=5, purge_gap=12):
    """
    Expanding-window walk-forward cross-validation with embargo / purge gap.

    For each fold produced by TimeSeriesSplit (operating on *unique sorted
    timestamps*) we:
      1. Remove `purge_gap` timestamps from the END of the training window
         to prevent triple-barrier target leakage across the boundary.
      2. Train a fresh LightGBM on the purged training set.
      3. Predict on the test set and accumulate OOS predictions.

    Returns
    -------
    oos_metrics : dict   — honest out-of-sample metrics over all folds
    fold_details : list  — per-fold diagnostic summaries
    median_best_iter : int — median best_iteration across folds (useful for
                             choosing n_estimators for the final production model)
    """
    logger.info("=" * 72)
    logger.info("Walk-Forward Validation | n_splits=%s | purge_gap=%s timestamps", n_splits, purge_gap)
    logger.info("=" * 72)

    # --- Unique sorted timestamp index for time-aware splitting -----------
    unique_ts = np.sort(dataset[TIMESTAMP_COLUMN].unique())
    n_timestamps = len(unique_ts)
    if n_timestamps < n_splits + 1:
        raise RuntimeError(
            f"Only {n_timestamps} unique timestamps — need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Accumulators for the single OOS vector
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_details = []
    best_iterations = []

    for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(tscv.split(unique_ts), start=1):
        # --- Resolve timestamp boundaries --------------------------------
        train_timestamps = unique_ts[train_ts_idx]
        test_timestamps = unique_ts[test_ts_idx]

        # Purge: remove `purge_gap` latest timestamps from train to create
        # an embargo zone that prevents triple-barrier label contamination.
        if purge_gap > 0 and len(train_timestamps) > purge_gap:
            train_timestamps = train_timestamps[:-purge_gap]
        elif purge_gap > 0:
            logger.warning(
                "Fold %s: purge_gap=%s >= train timestamps (%s), skipping purge",
                fold_idx, purge_gap, len(train_timestamps),
            )

        train_ts_set = set(train_timestamps)
        test_ts_set = set(test_timestamps)

        train_mask = dataset[TIMESTAMP_COLUMN].isin(train_ts_set)
        test_mask = dataset[TIMESTAMP_COLUMN].isin(test_ts_set)

        train_df = dataset.loc[train_mask].copy()
        test_df = dataset.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            logger.warning("Fold %s produced empty train or test — skipping", fold_idx)
            continue

        # Ensure both classes in training set
        train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())
        if len(train_classes) < 2:
            logger.warning("Fold %s has only class(es) %s in train — skipping", fold_idx, train_classes)
            continue

        # --- Feature clipping (fit on train, apply to train+test) --------
        fold_clip_bounds = build_feature_clip_bounds(train_df, feature_columns)
        train_df = apply_feature_clip_bounds(train_df, fold_clip_bounds)
        test_df = apply_feature_clip_bounds(test_df, fold_clip_bounds)

        x_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        x_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        # --- Train -------------------------------------------------------
        model = build_model(seed=seed)
        # Use last 20% of the train fold as an internal eval set for
        # early stopping, without contaminating the OOS test fold.
        internal_eval_size = max(1, int(len(x_train) * 0.2))
        x_fit = x_train.iloc[:-internal_eval_size]
        y_fit = y_train.iloc[:-internal_eval_size]
        x_eval = x_train.iloc[-internal_eval_size:]
        y_eval = y_train.iloc[-internal_eval_size:]

        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_eval, y_eval)],
            eval_metric="binary_logloss",
            categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),  # silent per-fold
            ],
        )

        best_iter = int(model.best_iteration_ or model.n_estimators_)
        best_iterations.append(best_iter)

        # --- Predict on OOS test fold ------------------------------------
        y_pred_fold = model.predict(x_test)
        y_proba_fold = model.predict_proba(x_test)

        all_y_true.append(y_test.values)
        all_y_pred.append(y_pred_fold)
        all_y_proba.append(y_proba_fold)

        # Per-fold quick summary
        fold_acc = float(accuracy_score(y_test, y_pred_fold))
        fold_auc = float(roc_auc_score(y_test, y_proba_fold[:, 1]))
        fold_info = {
            "fold": fold_idx,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "purged_timestamps": purge_gap,
            "train_period": {
                "start": str(train_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(train_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "test_period": {
                "start": str(test_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(test_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "best_iteration": best_iter,
            "accuracy": fold_acc,
            "roc_auc": fold_auc,
        }
        fold_details.append(fold_info)

        logger.info(
            "Fold %s/%s | train=%s rows [%s → %s] | test=%s rows [%s → %s] | "
            "best_iter=%s | acc=%.4f | auc=%.4f",
            fold_idx,
            n_splits,
            fold_info["train_rows"],
            fold_info["train_period"]["start"],
            fold_info["train_period"]["end"],
            fold_info["test_rows"],
            fold_info["test_period"]["start"],
            fold_info["test_period"]["end"],
            best_iter,
            fold_acc,
            fold_auc,
        )

    # --- Aggregate OOS vector ------------------------------------------------
    if not all_y_true:
        raise RuntimeError("All WFV folds were skipped — cannot compute OOS metrics.")

    oos_y_true = np.concatenate(all_y_true)
    oos_y_pred = np.concatenate(all_y_pred)
    oos_y_proba = np.vstack(all_y_proba)

    oos_metrics = evaluate_model(
        y_true=oos_y_true,
        y_pred=oos_y_pred,
        y_proba=oos_y_proba,
        split_name="oos",
        n_rows=len(oos_y_true),
    )

    median_best_iter = int(np.median(best_iterations))

    logger.info("-" * 72)
    logger.info(
        "OOS aggregate (%s folds, %s rows) | acc=%.4f | bal_acc=%.4f | "
        "f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
        len(fold_details),
        len(oos_y_true),
        oos_metrics["accuracy"],
        oos_metrics["balanced_accuracy"],
        oos_metrics["f1_macro"],
        oos_metrics["roc_auc"],
        oos_metrics["pr_auc"],
        oos_metrics["mcc"],
    )
    logger.info("Median best_iteration across folds: %s", median_best_iter)
    logger.info("=" * 72)

    return oos_metrics, fold_details, median_best_iter


# ═══════════════════════════════════════════════════════════════════════════
#  Production model (retrain on 100% of data)
# ═══════════════════════════════════════════════════════════════════════════

def train_production_model(dataset, feature_columns, seed, n_estimators):
    """
    Train the final deployment model on the ENTIRE dataset.

    The n_estimators is typically the median best_iteration from WFV,
    so we do NOT use early stopping here — every row is training data,
    and we have no hold-out to compute an eval metric on.
    """
    logger.info(
        "Training production model on 100%% of data (%s rows) with n_estimators=%s",
        len(dataset), n_estimators,
    )
    model = build_model(seed=seed, n_estimators=n_estimators)
    model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities — importance / persistence
# ═══════════════════════════════════════════════════════════════════════════

def build_period_payload(frame):
    if frame.empty:
        return None
    return {
        "start": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "end": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


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
        "task_type": "binary_directional",
        "train_period": build_period_payload(dataset),
        "wfv_n_splits": args.n_splits,
        "wfv_purge_gap": args.purge_gap,
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

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

        # ── Step 1: Walk-Forward Validation → honest OOS metrics ──────────
        oos_metrics, fold_details, median_best_iter = walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
        )

        # ── Step 2: Train production model on 100% of data ───────────────
        #    Clip bounds are computed on the FULL dataset because there is
        #    no hold-out anymore — this model sees everything we have.
        prod_clip_bounds = build_feature_clip_bounds(dataset, feature_columns)
        clipped_dataset = apply_feature_clip_bounds(dataset, prod_clip_bounds)

        if prod_clip_bounds:
            logger.info(
                "Feature clipping (production): %s numeric columns clipped to [%.2f%%, %.2f%%] quantiles",
                len(prod_clip_bounds),
                float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)) * 100,
                float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)) * 100,
            )

        prod_model = train_production_model(
            dataset=clipped_dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_estimators=median_best_iter,
        )

        # ── Step 3: Assemble final metrics payload & persist ─────────────
        metrics = {
            "oos_metrics": oos_metrics,
            "fold_details": fold_details,
            "median_best_iteration": median_best_iter,
            "total_rows": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "n_splits": args.n_splits,
            "purge_gap": args.purge_gap,
            "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            "candidate_rows": int(dataset.attrs.get("candidate_rows", len(dataset))),
            "excluded_by_event_filter_rows": int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
            "event_filter": dataset.attrs.get("event_filter_config"),
            "dataset_period": build_period_payload(dataset),
        }

        logger.info(
            "OOS metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | "
            "roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
            oos_metrics["accuracy"],
            oos_metrics["balanced_accuracy"],
            oos_metrics["f1_macro"],
            oos_metrics["roc_auc"],
            oos_metrics["pr_auc"],
            oos_metrics["mcc"],
        )

        log_feature_importance_ranking(prod_model, feature_columns)
        save_directional_artifacts(prod_model, metrics, dataset, feature_columns, prod_clip_bounds, args)

    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
