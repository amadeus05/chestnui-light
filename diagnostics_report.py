import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import config as cfg
import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run subset and per-symbol diagnostics for the directional model.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Universe to diagnose, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument("--purge-gap", type=int, default=12, help="Embargo gap in timestamps.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=max(0.55, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.55))),
        help="Threshold used for per-symbol signal diagnostics.",
    )
    parser.add_argument(
        "--top-k-symbols",
        type=int,
        default=6,
        help="Build a subset from the top-K symbols ranked by full-universe OOS accuracy.",
    )
    parser.add_argument(
        "--drop-bottom-n",
        type=int,
        default=2,
        help="Build a subset with the N weakest symbols removed based on full-universe OOS accuracy.",
    )
    parser.add_argument(
        "--output-path",
        default=str(cfg.MODELS_DIR / "diagnostics_report.json"),
        help="Path to the JSON report file.",
    )
    return parser.parse_args()


def run_walk_forward_with_predictions(dataset, feature_columns, seed, n_splits=5, purge_gap=12):
    unique_ts = np.sort(dataset[train.TIMESTAMP_COLUMN].unique())
    if len(unique_ts) < n_splits + 1:
        raise RuntimeError(
            f"Only {len(unique_ts)} unique timestamps - need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    best_iterations = []
    fold_details = []
    prediction_frames = []

    for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(tscv.split(unique_ts), start=1):
        train_timestamps = unique_ts[train_ts_idx]
        test_timestamps = unique_ts[test_ts_idx]

        if purge_gap > 0 and len(train_timestamps) > purge_gap:
            train_timestamps = train_timestamps[:-purge_gap]

        train_df = dataset.loc[dataset[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))].copy()
        test_df = dataset.loc[dataset[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))].copy()
        if train_df.empty or test_df.empty:
            continue

        if len(train_df[train.TARGET_COLUMN].unique()) < 2:
            continue

        clip_bounds = train.build_feature_clip_bounds(train_df, feature_columns)
        train_df = train.apply_feature_clip_bounds(train_df, clip_bounds)
        test_df = train.apply_feature_clip_bounds(test_df, clip_bounds)

        x_train = train_df[feature_columns]
        y_train = train_df[train.TARGET_COLUMN]
        x_test = test_df[feature_columns]
        y_test = test_df[train.TARGET_COLUMN]
        w_train = train.compute_sample_weights(train_df[train.TIMESTAMP_COLUMN])

        internal_eval_size = max(1, int(len(x_train) * 0.2))
        x_fit = x_train.iloc[:-internal_eval_size]
        y_fit = y_train.iloc[:-internal_eval_size]
        w_fit = w_train[:-internal_eval_size]
        x_eval = x_train.iloc[-internal_eval_size:]
        y_eval = y_train.iloc[-internal_eval_size:]

        model = train.build_model(seed=seed)
        model.fit(
            x_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(x_eval, y_eval)],
            eval_metric="binary_logloss",
            categorical_feature=[train.SYMBOL_COLUMN] if train.SYMBOL_COLUMN in feature_columns else "auto",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = int(model.best_iteration_ or model.n_estimators_)
        best_iterations.append(best_iter)

        y_pred_fold = model.predict(x_test)
        y_proba_fold = model.predict_proba(x_test)

        all_y_true.append(y_test.values)
        all_y_pred.append(y_pred_fold)
        all_y_proba.append(y_proba_fold)

        fold_details.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "best_iteration": best_iter,
                "accuracy": float(accuracy_score(y_test, y_pred_fold)),
                "roc_auc": float(roc_auc_score(y_test, y_proba_fold[:, 1])),
            }
        )

        fold_predictions = test_df[[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN, train.TARGET_COLUMN]].copy()
        fold_predictions["fold"] = fold_idx
        fold_predictions["pred"] = y_pred_fold
        fold_predictions["p_short"] = y_proba_fold[:, 0]
        fold_predictions["p_long"] = y_proba_fold[:, 1]
        prediction_frames.append(fold_predictions.reset_index(drop=True))

    if not all_y_true:
        raise RuntimeError("All WFV folds were skipped - cannot build diagnostics.")

    oos_y_true = np.concatenate(all_y_true)
    oos_y_pred = np.concatenate(all_y_pred)
    oos_y_proba = np.vstack(all_y_proba)
    oos_metrics = train.evaluate_model(
        y_true=oos_y_true,
        y_pred=oos_y_pred,
        y_proba=oos_y_proba,
        split_name="oos",
        n_rows=len(oos_y_true),
    )

    return {
        "oos_metrics": oos_metrics,
        "fold_details": fold_details,
        "median_best_iteration": int(np.median(best_iterations)),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def summarize_per_symbol(predictions: pd.DataFrame, threshold: float) -> list[dict]:
    rows = []
    signal_column = np.full(len(predictions), -1, dtype=int)
    signal_column[predictions["p_long"].to_numpy() >= threshold] = 1
    signal_column[predictions["p_short"].to_numpy() >= threshold] = 0
    prepared = predictions.copy()
    prepared["signal"] = signal_column

    for symbol, frame in prepared.groupby(train.SYMBOL_COLUMN):
        y_true = frame[train.TARGET_COLUMN].to_numpy()
        y_pred = frame["pred"].to_numpy()
        selected = frame.loc[frame["signal"] != -1].copy()

        summary = {
            "symbol": symbol,
            "rows": int(len(frame)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "majority_baseline": float(max((y_true == 0).mean(), (y_true == 1).mean())),
            "long_share": float((y_true == 1).mean()),
            "threshold": float(threshold),
            "threshold_rows": int(len(selected)),
            "threshold_coverage": float(len(selected) / len(frame)) if len(frame) else 0.0,
            "threshold_accuracy": None,
            "threshold_long_precision": None,
            "threshold_short_precision": None,
        }

        if len(np.unique(y_true)) == 2:
            summary["roc_auc"] = float(roc_auc_score(y_true, frame["p_long"]))
        else:
            summary["roc_auc"] = None

        if not selected.empty:
            selected_true = selected[train.TARGET_COLUMN].to_numpy()
            selected_pred = selected["signal"].to_numpy()
            summary["threshold_accuracy"] = float(accuracy_score(selected_true, selected_pred))

            long_signals = int((selected_pred == 1).sum())
            short_signals = int((selected_pred == 0).sum())
            long_tp = int(((selected_pred == 1) & (selected_true == 1)).sum())
            short_tp = int(((selected_pred == 0) & (selected_true == 0)).sum())

            if long_signals > 0:
                summary["threshold_long_precision"] = float(long_tp / long_signals)
            if short_signals > 0:
                summary["threshold_short_precision"] = float(short_tp / short_signals)

        rows.append(summary)

    rows.sort(key=lambda item: item["accuracy"])
    return rows


def build_subset_candidates(symbols: list[str], per_symbol_metrics: list[dict], top_k: int, drop_bottom_n: int) -> list[dict]:
    ranked_symbols = [row["symbol"] for row in sorted(per_symbol_metrics, key=lambda item: item["accuracy"], reverse=True)]
    weakest_symbols = [row["symbol"] for row in per_symbol_metrics[:drop_bottom_n]]

    subsets = [{"name": "all_configured", "symbols": list(symbols)}]

    if weakest_symbols and len(symbols) > len(weakest_symbols):
        subsets.append(
            {
                "name": f"drop_bottom_{len(weakest_symbols)}_by_symbol_accuracy",
                "symbols": [symbol for symbol in symbols if symbol not in set(weakest_symbols)],
            }
        )

    top_symbols = [symbol for symbol in ranked_symbols if symbol in set(symbols)][:top_k]
    if top_symbols and len(top_symbols) < len(symbols):
        subsets.append(
            {
                "name": f"top_{len(top_symbols)}_by_symbol_accuracy",
                "symbols": top_symbols,
            }
        )

    unique_subsets = []
    seen = set()
    for subset in subsets:
        key = tuple(subset["symbols"])
        if key in seen:
            continue
        seen.add(key)
        unique_subsets.append(subset)
    return unique_subsets


def evaluate_subset(name: str, symbols: list[str], args) -> dict:
    dataset = train.load_training_frame(args.db_path, symbols)
    feature_columns = train.select_feature_columns(dataset)
    diagnostics = run_walk_forward_with_predictions(
        dataset=dataset,
        feature_columns=feature_columns,
        seed=args.seed,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
    )
    threshold_key = f"{args.confidence_threshold:.2f}"
    threshold_metrics = diagnostics["oos_metrics"]["probability_threshold_metrics"].get(threshold_key, {})

    return {
        "name": name,
        "symbols": list(symbols),
        "rows": int(len(dataset)),
        "feature_count": int(len(feature_columns)),
        "median_best_iteration": diagnostics["median_best_iteration"],
        "accuracy": float(diagnostics["oos_metrics"]["accuracy"]),
        "balanced_accuracy": float(diagnostics["oos_metrics"]["balanced_accuracy"]),
        "roc_auc": float(diagnostics["oos_metrics"]["roc_auc"]),
        "mcc": float(diagnostics["oos_metrics"]["mcc"]),
        "threshold": float(args.confidence_threshold),
        "threshold_accuracy": threshold_metrics.get("signal_accuracy"),
        "threshold_coverage": threshold_metrics.get("coverage"),
    }


def main():
    args = parse_args()
    threshold_key = f"{args.confidence_threshold:.2f}"

    dataset = train.load_training_frame(args.db_path, args.symbols)
    feature_columns = train.select_feature_columns(dataset)
    diagnostics = run_walk_forward_with_predictions(
        dataset=dataset,
        feature_columns=feature_columns,
        seed=args.seed,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
    )
    per_symbol_metrics = summarize_per_symbol(diagnostics["predictions"], args.confidence_threshold)
    subset_candidates = build_subset_candidates(args.symbols, per_symbol_metrics, args.top_k_symbols, args.drop_bottom_n)
    subset_report = [evaluate_subset(subset["name"], subset["symbols"], args) for subset in subset_candidates]

    payload = {
        "symbols": list(args.symbols),
        "event_filter": dataset.attrs.get("event_filter_config"),
        "rows": int(len(dataset)),
        "feature_count": int(len(feature_columns)),
        "n_splits": args.n_splits,
        "purge_gap": args.purge_gap,
        "confidence_threshold": float(args.confidence_threshold),
        "candidate_rows": int(dataset.attrs.get("candidate_rows", len(dataset))),
        "excluded_by_event_filter_rows": int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
        "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
        "full_universe_metrics": {
            "accuracy": float(diagnostics["oos_metrics"]["accuracy"]),
            "balanced_accuracy": float(diagnostics["oos_metrics"]["balanced_accuracy"]),
            "roc_auc": float(diagnostics["oos_metrics"]["roc_auc"]),
            "mcc": float(diagnostics["oos_metrics"]["mcc"]),
            "threshold_metrics": diagnostics["oos_metrics"]["probability_threshold_metrics"].get(threshold_key),
        },
        "subset_report": subset_report,
        "per_symbol_report": per_symbol_metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("Diagnostics report saved to %s", output_path)
    logger.info(
        "Full universe | rows=%s | acc=%.4f | auc=%.4f | threshold=%s acc=%s cov=%s",
        payload["rows"],
        payload["full_universe_metrics"]["accuracy"],
        payload["full_universe_metrics"]["roc_auc"],
        threshold_key,
        payload["full_universe_metrics"]["threshold_metrics"].get("signal_accuracy")
        if payload["full_universe_metrics"]["threshold_metrics"]
        else None,
        payload["full_universe_metrics"]["threshold_metrics"].get("coverage")
        if payload["full_universe_metrics"]["threshold_metrics"]
        else None,
    )


if __name__ == "__main__":
    main()
