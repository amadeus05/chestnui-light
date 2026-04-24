import logging

import config as cfg

from .artifacts import log_feature_importance_ranking, save_directional_artifacts
from .config_snapshot import build_experiment_snapshot
from .cli import parse_args
from .dataset import load_training_frame
from .diagnostics import build_dataset_diagnostics, build_fold_stability_payload, build_period_payload
from .features import apply_feature_clip_bounds, build_feature_clip_bounds, select_feature_columns
from .history import (
    build_train_history_entry,
    get_train_history_path,
    log_train_history_summary,
    save_train_history,
)
from .importance import build_fold_importance_summary
from .log import logger
from .metrics import format_optional_metric
from .production import train_production_model
from .walk_forward import walk_forward_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    try:
        args = parse_args()
        experiment_snapshot = build_experiment_snapshot()
        dataset = load_training_frame(args.db_path, args.symbols)
        feature_columns = select_feature_columns(dataset)

        logger.info("Loaded %s rows with %s features", len(dataset), len(feature_columns))
        logger.info("Using symbols: %s", ", ".join(args.symbols))
        logger.info(
            "Experiment=%s | labeling_profile=%s | training_profile=%s",
            experiment_snapshot["experiment"],
            experiment_snapshot["labeling_profile"],
            experiment_snapshot["training_profile"],
        )
        logger.info(
            "Labeling config: horizon=%s | dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
            experiment_snapshot["labeling"]["horizon"],
            experiment_snapshot["labeling"]["use_dynamic_barriers"],
            experiment_snapshot["labeling"]["barrier_min_pct"],
            experiment_snapshot["labeling"]["barrier_max_pct"],
            experiment_snapshot["labeling"]["barrier_tp_to_sl_ratio"],
        )
        logger.info(
            "Training config: disabled_features=%s | clip=%s [%.2f%%, %.2f%%]",
            len(experiment_snapshot["training"]["disabled_feature_columns"]),
            experiment_snapshot["training"]["feature_clip_enabled"],
            experiment_snapshot["training"]["feature_clip_lower_q"] * 100,
            experiment_snapshot["training"]["feature_clip_upper_q"] * 100,
        )
        logger.info(
            "Candidate universe: kept %s rows after deterministic event filter, excluded %s rows",
            int(dataset.attrs.get("candidate_rows", len(dataset))),
            int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
        )
        logger.info(
            "Directional baseline inside candidate universe: excluded %s non-directional rows with Target=0 before split",
            int(dataset.attrs.get("excluded_non_directional_rows", 0)),
        )
        timeline_profile = dataset.attrs.get("all_timestamps_profile", {})
        logger.info(
            "Dataset timeline | all_timestamps=%s [%s -> %s] | rows before filter=%s | candidates=%s | directional=%s",
            int(timeline_profile.get("count", 0)),
            timeline_profile.get("first"),
            timeline_profile.get("last"),
            int(dataset.attrs.get("candidate_rows_before_filter", len(dataset))),
            int(dataset.attrs.get("candidate_rows", len(dataset))),
            len(dataset),
        )
        feature_rows_by_symbol = dataset.attrs.get("feature_table_row_counts_by_symbol", {})
        if feature_rows_by_symbol:
            logger.info(
                "Feature table rows by symbol: %s",
                ", ".join(
                    f"{symbol}={profile.get('rows', 0)}"
                    for symbol, profile in sorted(feature_rows_by_symbol.items())
                ),
            )

        # ── Step 1: Walk-Forward Validation → honest OOS metrics ──────────
        oos_metrics, fold_details, median_best_iter, fold_importance = walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
            split_mode=args.split_mode,
            monthly_train_months=args.monthly_train_months,
            monthly_test_months=args.monthly_test_months,
            monthly_window_mode=args.monthly_window_mode,
        )
        dataset_diagnostics = build_dataset_diagnostics(dataset, fold_details)

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
            "fold_stability": build_fold_stability_payload(fold_details),
            "fold_feature_importance_top": build_fold_importance_summary(fold_importance, limit=20),
            "median_best_iteration": median_best_iter,
            "total_rows": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "n_splits": args.n_splits,
            "requested_purge_gap": args.purge_gap,
            "purge_gap": int(fold_details[0]["purged_timestamps"]) if fold_details else args.purge_gap,
            "split_mode": args.split_mode,
            "monthly_train_months": args.monthly_train_months,
            "monthly_test_months": args.monthly_test_months,
            "monthly_window_mode": args.monthly_window_mode,
            "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            "candidate_rows": int(dataset.attrs.get("candidate_rows", len(dataset))),
            "excluded_by_event_filter_rows": int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
            "all_timestamps_count": dataset_diagnostics["all_timestamps_count"],
            "first_all_timestamp": dataset_diagnostics["first_all_timestamp"],
            "last_all_timestamp": dataset_diagnostics["last_all_timestamp"],
            "candidate_rows_before_filter": dataset_diagnostics["candidate_rows_before_filter"],
            "candidate_rows_after_filter": dataset_diagnostics["candidate_rows_after_filter"],
            "directional_rows_after_filter": dataset_diagnostics["directional_rows_after_filter"],
            "fold_boundary_timestamps": dataset_diagnostics["fold_boundary_timestamps"],
            "feature_table_row_counts_by_symbol": dataset_diagnostics["feature_table_row_counts_by_symbol"],
            "dataset_diagnostics": dataset_diagnostics,
            "event_filter": dataset.attrs.get("event_filter_config"),
            "experiment": experiment_snapshot,
            "dataset_period": build_period_payload(dataset),
        }

        logger.info(
            "OOS metrics | accuracy=%.4f | balanced_accuracy=%s | f1_macro=%.4f | "
            "roc_auc=%s | pr_auc=%s | mcc=%s",
            oos_metrics["accuracy"],
            format_optional_metric(oos_metrics["balanced_accuracy"]),
            oos_metrics["f1_macro"],
            format_optional_metric(oos_metrics["roc_auc"]),
            format_optional_metric(oos_metrics["pr_auc"]),
            format_optional_metric(oos_metrics["mcc"]),
        )

        log_feature_importance_ranking(prod_model, feature_columns)
        save_directional_artifacts(
            prod_model,
            metrics,
            dataset,
            feature_columns,
            prod_clip_bounds,
            fold_importance,
            args,
            experiment_snapshot,
        )
        history_path = get_train_history_path(args.model_name)
        history_entry = build_train_history_entry(args, metrics, experiment_snapshot)
        history = save_train_history(history_path, history_entry)
        logger.info("Saved train history to %s", history_path)
        log_train_history_summary(history_entry, history, limit=10)

    except Exception as exc:
        logger.exception("%s", exc)
        raise SystemExit(1) from exc
