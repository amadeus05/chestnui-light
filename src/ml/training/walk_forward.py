import numpy as np
import pandas as pd
import config as cfg
from sklearn.metrics import accuracy_score

from .constants import TARGET_COLUMN, TIMESTAMP_COLUMN
from .dataset import format_timestamp
from .features import apply_feature_clip_bounds, build_feature_clip_bounds
from .importance import aggregate_fold_importance, build_fold_importance_frame
from .log import logger
from .metrics import evaluate_model, format_optional_metric, safe_roc_auc_score
from .model_lgb import compute_sample_weights, fit_model_with_internal_eval
from .splits import build_timestamp_splits


def resolve_effective_purge_gap(requested_purge_gap):
    base_horizon = max(1, int(getattr(cfg, "HORIZON", 1)))
    label_horizon = base_horizon
    if bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)):
        label_horizon = max(label_horizon, int(getattr(cfg, "ADAPTIVE_HORIZON_MAX", base_horizon)))
    return max(int(requested_purge_gap), label_horizon), label_horizon

def walk_forward_validation(
    dataset,
    feature_columns,
    seed,
    n_splits=5,
    purge_gap=12,
    split_mode="tscv",
    monthly_train_months=6,
    monthly_test_months=1,
    monthly_window_mode="expanding",
):
    """
    Expanding-window walk-forward cross-validation with embargo / purge gap.

    For each fold from ``build_timestamp_splits`` (time-ordered unique
    timestamps) we:
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
    effective_purge_gap, label_horizon = resolve_effective_purge_gap(purge_gap)
    if effective_purge_gap != int(purge_gap):
        logger.warning(
            "Increasing purge_gap from %s to %s to cover max label horizon=%s",
            purge_gap,
            effective_purge_gap,
            label_horizon,
        )

    logger.info("=" * 72)
    logger.info(
        "Walk-Forward Validation | split_mode=%s | n_splits=%s | monthly_train=%s | "
        "monthly_test=%s | monthly_window=%s | purge_gap=%s timestamps",
        split_mode,
        n_splits,
        monthly_train_months,
        monthly_test_months,
        monthly_window_mode,
        effective_purge_gap,
    )
    logger.info("=" * 72)

    # --- Unique sorted timestamp index for time-aware splitting -----------
    # Use the full bar timeline when available. Event-filtered/directional
    # rows can be sparse, so purging N selected timestamps is not the same as
    # purging N market bars near the fold boundary.
    unique_ts = np.asarray(dataset.attrs.get("all_timestamps", np.sort(dataset[TIMESTAMP_COLUMN].unique())))
    n_timestamps = len(unique_ts)
    if n_timestamps < n_splits + 1:
        raise RuntimeError(
            f"Only {n_timestamps} unique timestamps — need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    timestamp_splits = build_timestamp_splits(
        unique_ts=unique_ts,
        n_splits=n_splits,
        split_mode=split_mode,
        monthly_train_months=monthly_train_months,
        monthly_test_months=monthly_test_months,
        monthly_window_mode=monthly_window_mode,
    )
    if not timestamp_splits:
        raise RuntimeError("No walk-forward timestamp splits were produced.")

    # Accumulators for the single OOS vector
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_details = []
    best_iterations = []
    fold_importance_frames = []

    for fold_idx, original_train_timestamps, test_timestamps in timestamp_splits:
        # --- Resolve timestamp boundaries --------------------------------
        train_timestamps = original_train_timestamps

        # Purge: remove `purge_gap` latest timestamps from train to create
        # an embargo zone that prevents triple-barrier label contamination.
        if effective_purge_gap > 0 and len(train_timestamps) > effective_purge_gap:
            train_timestamps = train_timestamps[:-effective_purge_gap]
        elif effective_purge_gap > 0:
            logger.warning(
                "Fold %s: purge_gap=%s >= train timestamps (%s), skipping purge",
                fold_idx, effective_purge_gap, len(train_timestamps),
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
        w_train = compute_sample_weights(train_df)

        # --- Train -------------------------------------------------------
        model, fit_metadata = fit_model_with_internal_eval(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            feature_columns=feature_columns,
            seed=seed,
            best_iterations_so_far=best_iterations,
        )

        eval_plan = fit_metadata["eval_plan"]
        internal_eval_size = int(fit_metadata["internal_eval_size"])
        best_iter = int(fit_metadata["best_iter"])
        best_iterations.append(best_iter)
        fold_importance_frames.append(build_fold_importance_frame(model, feature_columns, fold_idx))

        # --- Predict on OOS test fold ------------------------------------
        y_pred_fold = model.predict(x_test)
        y_proba_fold = model.predict_proba(x_test)

        all_y_true.append(y_test.values)
        all_y_pred.append(y_pred_fold)
        all_y_proba.append(y_proba_fold)

        # Per-fold quick summary
        fold_acc = float(accuracy_score(y_test, y_pred_fold))
        fold_auc = safe_roc_auc_score(y_test, y_proba_fold[:, 1])
        fold_info = {
            "fold": fold_idx,
            "split_mode": split_mode,
            "monthly_window_mode": monthly_window_mode if split_mode == "monthly" else None,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "requested_purge_gap": int(purge_gap),
            "purged_timestamps": int(effective_purge_gap),
            "max_label_horizon": int(label_horizon),
            "timestamp_boundaries": {
                "train_start": format_timestamp(train_timestamps[0]),
                "train_end_before_purge": format_timestamp(original_train_timestamps[-1]),
                "train_end_after_purge": format_timestamp(train_timestamps[-1]),
                "test_start": format_timestamp(test_timestamps[0]),
                "test_end": format_timestamp(test_timestamps[-1]),
                "train_unique_timestamps_after_purge": int(len(train_timestamps)),
                "test_unique_timestamps": int(len(test_timestamps)),
            },
            "train_period": {
                "start": str(train_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(train_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "test_period": {
                "start": str(test_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(test_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "internal_eval": {
                "rows": int(internal_eval_size),
                "fraction": float(eval_plan["eval_fraction"]),
                "fit_positive_rate": float(eval_plan["fit_rate"]),
                "eval_positive_rate": float(eval_plan["eval_rate"]),
                "was_expanded": bool(eval_plan["was_expanded"]),
                "fallback_used": bool(fit_metadata["fallback_used"]),
                "fallback_n_estimators": (
                    int(fit_metadata["fallback_n_estimators"])
                    if fit_metadata["fallback_n_estimators"] is not None
                    else None
                ),
            },
            "best_iteration": best_iter,
            "accuracy": fold_acc,
            "roc_auc": fold_auc,
        }
        fold_details.append(fold_info)

        logger.info(
            "Fold %s/%s | train=%s rows [%s → %s] | test=%s rows [%s → %s] | "
            "eval=%s (fit_pos=%.3f eval_pos=%.3f%s%s) | best_iter=%s | acc=%.4f | auc=%s",
            fold_idx,
            len(timestamp_splits),
            fold_info["train_rows"],
            fold_info["train_period"]["start"],
            fold_info["train_period"]["end"],
            fold_info["test_rows"],
            fold_info["test_period"]["start"],
            fold_info["test_period"]["end"],
            internal_eval_size,
            float(eval_plan["fit_rate"]),
            float(eval_plan["eval_rate"]),
            ", expanded" if eval_plan["was_expanded"] else "",
            f", fallback={fit_metadata['fallback_n_estimators']}" if fit_metadata["fallback_used"] else "",
            best_iter,
            fold_acc,
            format_optional_metric(fold_auc),
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
        "OOS aggregate (%s folds, %s rows) | acc=%.4f | bal_acc=%s | "
        "f1_macro=%.4f | roc_auc=%s | pr_auc=%s | mcc=%s",
        len(fold_details),
        len(oos_y_true),
        oos_metrics["accuracy"],
        format_optional_metric(oos_metrics["balanced_accuracy"]),
        oos_metrics["f1_macro"],
        format_optional_metric(oos_metrics["roc_auc"]),
        format_optional_metric(oos_metrics["pr_auc"]),
        format_optional_metric(oos_metrics["mcc"]),
    )
    logger.info("Median best_iteration across folds: %s", median_best_iter)
    logger.info("=" * 72)

    fold_importance = aggregate_fold_importance(fold_importance_frames, feature_columns)
    if not fold_importance.empty:
        logger.info("Fold feature importance top by mean gain:")
        for rank, row in enumerate(fold_importance.head(10).itertuples(index=False), start=1):
            logger.info(
                "%s. %s | mean_gain=%.6f | std_gain=%.6f | top10_folds=%s | top20_folds=%s",
                rank,
                row.feature,
                float(row.mean_gain_by_fold),
                float(row.std_gain_by_fold),
                int(row.top_10_fold_count),
                int(row.top_20_fold_count),
            )

    return oos_metrics, fold_details, median_best_iter, fold_importance
