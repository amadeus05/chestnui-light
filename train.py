import argparse
import json
import logging
from datetime import datetime, timezone

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


def build_experiment_snapshot() -> dict:
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "labeling": {
            "horizon": int(getattr(cfg, "HORIZON", 0)),
            "tp_pct": float(getattr(cfg, "TP_PCT", 0.0)),
            "sl_pct": float(getattr(cfg, "SL_PCT", 0.0)),
            "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
            "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
            "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
            "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
            "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
            "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
        },
        "training": {
            "disabled_feature_columns": sorted(getattr(cfg, "MANUAL_DISABLED_FEATURE_COLUMNS", [])),
            "feature_clip_enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "feature_clip_lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.0)),
            "feature_clip_upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 1.0)),
        },
    }


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

def compute_sample_weights(timestamps: pd.Series, half_life_days: float = 365.0) -> np.ndarray:
    ts = pd.to_datetime(timestamps)
    days_ago = (ts.max() - ts).dt.total_seconds() / 86400.0
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago.values)
    return weights


def build_model(seed, n_estimators=800):
    """Instantiate LightGBM binary classifier. class_weight=None → calibrated probs."""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.005,
        num_leaves=15,
        min_child_samples=150,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_split_gain=0.01,
        subsample_freq=1,
    )


def build_meta_model(seed, n_estimators=300):
    """Smaller regression model that estimates expected return of a base signal."""
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=80,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_split_gain=0.0,
        subsample_freq=1,
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
    confidence_thresholds = get_confidence_thresholds()
    probability_threshold_metrics = {}
    p_short = y_proba[:, 0]
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    for threshold in confidence_thresholds:
        threshold = float(threshold)
        signal = build_directional_signal_from_probabilities(
            proba_long=p_long,
            proba_short=p_short,
            threshold=threshold,
            n_rows=n_rows,
        )
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


def get_confidence_thresholds():
    return sorted(
        {
            round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5))), 2),
            0.55,
            0.60,
            0.65,
            0.70,
        }
    )


def build_directional_signal_from_probabilities(proba_long, proba_short, threshold, n_rows):
    signal = np.full(n_rows, -1, dtype=int)
    signal[np.asarray(proba_long) >= threshold] = 1
    signal[np.asarray(proba_short) >= threshold] = 0
    return signal


def evaluate_signal_subset(y_true, y_pred, selected_mask, realized_return_pct=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    realized_return_pct = None if realized_return_pct is None else np.asarray(realized_return_pct, dtype=float)

    rows = int(selected_mask.sum())
    if rows == 0:
        return {
            "rows": 0,
            "coverage": 0.0,
            "win_rate": None,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_macro": None,
            "mcc": None,
            "net_pnl_pct": None,
            "avg_pnl_pct": None,
        }

    subset_y_true = y_true[selected_mask]
    subset_y_pred = y_pred[selected_mask]
    wins = int((subset_y_true == subset_y_pred).sum())
    losses = int(rows - wins)
    subset_realized_return_pct = (
        realized_return_pct[selected_mask] if realized_return_pct is not None else np.where(subset_y_true == subset_y_pred, 1.0, -1.0)
    )
    return {
        "rows": rows,
        "coverage": float(rows / len(y_true)) if len(y_true) else 0.0,
        "win_rate": float(wins / rows) if rows else None,
        "wins": wins,
        "losses": losses,
        "accuracy": float(accuracy_score(subset_y_true, subset_y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(subset_y_true, subset_y_pred)),
        "f1_macro": float(f1_score(subset_y_true, subset_y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(subset_y_true, subset_y_pred)),
        "net_pnl_pct": float(np.sum(subset_realized_return_pct)),
        "avg_pnl_pct": float(np.mean(subset_realized_return_pct)),
    }


def build_meta_feature_frame(frame, feature_columns):
    meta_frame = frame[feature_columns].copy()
    meta_frame["meta_proba_long"] = frame["proba_long"].astype(float)
    meta_frame["meta_proba_short"] = frame["proba_short"].astype(float)
    meta_frame["meta_confidence"] = frame["base_confidence"].astype(float)
    meta_frame["meta_margin"] = frame["base_margin"].astype(float)
    meta_frame["meta_pred_side"] = frame["base_pred"].astype(int)
    meta_frame["meta_signed_confidence"] = np.where(
        frame["base_pred"].astype(int) == 1,
        frame["base_confidence"].astype(float),
        -frame["base_confidence"].astype(float),
    )
    return meta_frame


def fit_meta_model(meta_train_df, meta_feature_columns, seed):
    meta_model = build_meta_model(seed=seed)
    x_meta = build_meta_feature_frame(meta_train_df, meta_feature_columns)
    y_meta = meta_train_df["realized_return_pct"].astype(float)
    w_meta = compute_sample_weights(meta_train_df[TIMESTAMP_COLUMN])

    if len(x_meta) >= 100 and y_meta.nunique() > 10:
        eval_size = max(20, int(len(x_meta) * 0.2))
        if len(x_meta) - eval_size >= 50:
            x_fit = x_meta.iloc[:-eval_size]
            y_fit = y_meta.iloc[:-eval_size]
            w_fit = w_meta[:-eval_size]
            x_eval = x_meta.iloc[-eval_size:]
            y_eval = y_meta.iloc[-eval_size:]
            meta_model.fit(
                x_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(x_eval, y_eval)],
                eval_metric="l2",
                categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in meta_feature_columns else "auto",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            return meta_model

    meta_model.fit(
        x_meta,
        y_meta,
        sample_weight=w_meta,
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in meta_feature_columns else "auto",
    )
    return meta_model


def get_roundtrip_cost_pct():
    taker = float(getattr(cfg, "TAKER_COM", 0.0))
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0))
    return 2.0 * (taker + slippage)


def evaluate_meta_filter_comparison(oos_prediction_frame, feature_columns, seed):
    if oos_prediction_frame.empty:
        return None

    fold_ids = sorted(int(value) for value in oos_prediction_frame["fold"].dropna().unique())
    if len(fold_ids) < 2:
        return None

    meta_feature_columns = list(feature_columns)

    base_thresholds = get_confidence_thresholds()
    meta_thresholds = sorted(
        {
            0.000,
            0.001,
            0.002,
            0.004,
            0.006,
        }
    )

    per_base_threshold = {}
    combo_rows = []
    meta_rows_by_base_threshold = {}
    roundtrip_cost_pct = get_roundtrip_cost_pct()
    for base_threshold in base_thresholds:
        meta_prediction_frames = []
        prior_meta_frames = []

        for fold_id in fold_ids:
            fold_frame = oos_prediction_frame.loc[oos_prediction_frame["fold"] == fold_id].copy()
            base_signal = build_directional_signal_from_probabilities(
                proba_long=fold_frame["proba_long"].astype(float).to_numpy(),
                proba_short=fold_frame["proba_short"].astype(float).to_numpy(),
                threshold=float(base_threshold),
                n_rows=len(fold_frame),
            )
            selected_mask = base_signal != -1
            selected_fold_frame = fold_frame.loc[selected_mask].copy()
            if selected_fold_frame.empty:
                continue

            selected_fold_frame["base_signal"] = base_signal[selected_mask]
            selected_fold_frame["realized_return_pct"] = np.where(
                selected_fold_frame["base_signal"].astype(int) == selected_fold_frame["y_true"].astype(int),
                selected_fold_frame["barrier_take_pct"].astype(float) - roundtrip_cost_pct,
                -selected_fold_frame["barrier_stop_pct"].astype(float) - roundtrip_cost_pct,
            )

            prior_train = pd.concat(prior_meta_frames, ignore_index=True) if prior_meta_frames else pd.DataFrame()
            if not prior_train.empty and prior_train["realized_return_pct"].nunique() > 10 and len(prior_train) >= 200:
                meta_model = fit_meta_model(prior_train, meta_feature_columns, seed)
                x_fold_meta = build_meta_feature_frame(selected_fold_frame, meta_feature_columns)
                selected_fold_frame["meta_expected_return_pct"] = meta_model.predict(x_fold_meta)
                meta_prediction_frames.append(selected_fold_frame)

            prior_meta_frames.append(selected_fold_frame)

        if not meta_prediction_frames:
            continue

        meta_eval_df = pd.concat(meta_prediction_frames, ignore_index=True).sort_values(
            [TIMESTAMP_COLUMN, SYMBOL_COLUMN]
        ).reset_index(drop=True)
        meta_rows_by_base_threshold[f"{float(base_threshold):.2f}"] = int(len(meta_eval_df))

        y_true = meta_eval_df["y_true"].astype(int).to_numpy()
        base_signal = meta_eval_df["base_signal"].astype(int).to_numpy()
        meta_expected_return_pct = meta_eval_df["meta_expected_return_pct"].astype(float).to_numpy()
        realized_return_pct = meta_eval_df["realized_return_pct"].astype(float).to_numpy()

        base_reference = evaluate_signal_subset(
            y_true=y_true,
            y_pred=base_signal,
            selected_mask=np.ones(len(meta_eval_df), dtype=bool),
            realized_return_pct=realized_return_pct,
        )
        base_reference["threshold"] = float(base_threshold)
        per_base_threshold[f"{float(base_threshold):.2f}"] = {
            "base_only": base_reference,
            "meta_variants": {},
        }

        for meta_threshold in meta_thresholds:
            combo_selected_mask = meta_expected_return_pct >= float(meta_threshold)
            combo_metrics = evaluate_signal_subset(
                y_true=y_true,
                y_pred=base_signal,
                selected_mask=combo_selected_mask,
                realized_return_pct=realized_return_pct,
            )
            combo_metrics["base_threshold"] = float(base_threshold)
            combo_metrics["meta_threshold"] = float(meta_threshold)
            combo_metrics["delta_wins_vs_base"] = int(combo_metrics["wins"] - base_reference["wins"])
            combo_metrics["delta_losses_vs_base"] = int(combo_metrics["losses"] - base_reference["losses"])
            combo_metrics["delta_net_pnl_pct_vs_base"] = (
                float(combo_metrics["net_pnl_pct"] - base_reference["net_pnl_pct"])
                if combo_metrics["net_pnl_pct"] is not None and base_reference["net_pnl_pct"] is not None
                else None
            )
            combo_metrics["delta_win_rate_vs_base"] = (
                float(combo_metrics["win_rate"] - base_reference["win_rate"])
                if combo_metrics["win_rate"] is not None and base_reference["win_rate"] is not None
                else None
            )
            per_base_threshold[f"{float(base_threshold):.2f}"]["meta_variants"][f"{float(meta_threshold):.2f}"] = combo_metrics
            combo_rows.append(combo_metrics)

    default_base_threshold = round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.55))), 2)
    default_base_key = f"{default_base_threshold:.2f}"
    if default_base_key not in per_base_threshold:
        return None
    base_reference = per_base_threshold[default_base_key]["base_only"]
    default_variants = per_base_threshold[default_base_key]["meta_variants"]

    non_empty_combos = [row for row in combo_rows if row["rows"] > 0]
    default_non_empty = [row for row in default_variants.values() if row["rows"] > 0]
    best_default_by_wins = max(default_non_empty, key=lambda item: (item["wins"], item["win_rate"] or 0.0), default=None)
    min_rows_for_quality = max(100, int(meta_rows_by_base_threshold[default_base_key] * 0.05))
    best_default_by_net_pnl = max(
        default_non_empty,
        key=lambda item: (item["net_pnl_pct"] if item["net_pnl_pct"] is not None else float("-inf"), item["wins"]),
        default=None,
    )
    best_default_by_win_rate = max(
        [row for row in default_non_empty if row["rows"] >= min_rows_for_quality],
        key=lambda item: (item["win_rate"] or 0.0, item["wins"]),
        default=None,
    )
    best_overall_by_wins = max(non_empty_combos, key=lambda item: (item["wins"], item["win_rate"] or 0.0), default=None)

    return {
        "rows_with_meta_oos": int(meta_rows_by_base_threshold.get(default_base_key, 0)),
        "rows_with_meta_oos_by_base_threshold": meta_rows_by_base_threshold,
        "folds_with_meta_oos": fold_ids[1:],
        "base_thresholds": [float(value) for value in base_thresholds],
        "meta_thresholds": [float(value) for value in meta_thresholds],
        "default_base_threshold": float(default_base_threshold),
        "roundtrip_cost_pct": float(roundtrip_cost_pct),
        "base_reference": base_reference,
        "per_base_threshold": per_base_threshold,
        "best_default_by_wins": best_default_by_wins,
        "best_default_by_net_pnl": best_default_by_net_pnl,
        "best_default_by_win_rate": best_default_by_win_rate,
        "best_overall_by_wins": best_overall_by_wins,
    }


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
    oos_prediction_frames = []

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
        w_train = compute_sample_weights(train_df[TIMESTAMP_COLUMN])

        # --- Train -------------------------------------------------------
        model = build_model(seed=seed)
        # Use last 15% of the train fold as an internal eval set for
        # early stopping, without contaminating the OOS test fold.
        internal_eval_size = max(1, int(len(x_train) * 0.15))
        x_fit = x_train.iloc[:-internal_eval_size]
        y_fit = y_train.iloc[:-internal_eval_size]
        w_fit = w_train[:-internal_eval_size]
        x_eval = x_train.iloc[-internal_eval_size:]
        y_eval = y_train.iloc[-internal_eval_size:]

        model.fit(
            x_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(x_eval, y_eval)],
            eval_metric="binary_logloss",
            categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
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

        prediction_frame = test_df[
            [TIMESTAMP_COLUMN, SYMBOL_COLUMN, "barrier_take_pct", "barrier_stop_pct", *feature_columns]
        ].copy()
        prediction_frame["fold"] = fold_idx
        prediction_frame["y_true"] = y_test.values
        prediction_frame["base_pred"] = y_pred_fold
        prediction_frame["proba_short"] = y_proba_fold[:, 0]
        prediction_frame["proba_long"] = y_proba_fold[:, 1]
        prediction_frame["base_confidence"] = np.max(y_proba_fold, axis=1)
        prediction_frame["base_margin"] = np.abs(y_proba_fold[:, 1] - y_proba_fold[:, 0])
        prediction_frame["base_correct"] = (prediction_frame["base_pred"] == prediction_frame["y_true"]).astype(int)
        oos_prediction_frames.append(prediction_frame)

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

    oos_prediction_frame = pd.concat(oos_prediction_frames, ignore_index=True)
    return oos_metrics, fold_details, median_best_iter, oos_prediction_frame


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
    w_prod = compute_sample_weights(dataset[TIMESTAMP_COLUMN])
    model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        sample_weight=w_prod,
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


def build_fold_stability_payload(fold_details):
    if not fold_details:
        return {
            "accuracy_std": None,
            "accuracy_range": None,
            "roc_auc_std": None,
            "roc_auc_range": None,
        }

    accuracy_values = np.asarray([float(fold["accuracy"]) for fold in fold_details], dtype=float)
    roc_auc_values = np.asarray([float(fold["roc_auc"]) for fold in fold_details], dtype=float)
    return {
        "accuracy_std": float(np.std(accuracy_values)),
        "accuracy_range": float(np.max(accuracy_values) - np.min(accuracy_values)),
        "roc_auc_std": float(np.std(roc_auc_values)),
        "roc_auc_range": float(np.max(roc_auc_values) - np.min(roc_auc_values)),
    }


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


def build_train_history_entry(args, metrics, experiment_snapshot):
    oos_metrics = metrics["oos_metrics"]
    fold_stability = build_fold_stability_payload(metrics.get("fold_details", []))
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
    rows = [
        ("Accuracy", format_compact_metric_value(history_entry["accuracy"] * 100, percent=True, decimals=2)),
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


def build_meta_filter_summary_lines(meta_payload):
    if not meta_payload:
        return ["Meta-filter comparison unavailable."]

    base_reference = meta_payload.get("base_reference") or {}
    best_default_by_wins = meta_payload.get("best_default_by_wins")
    best_default_by_net_pnl = meta_payload.get("best_default_by_net_pnl")
    best_default_by_win_rate = meta_payload.get("best_default_by_win_rate")

    rows = [
        (
            "Base only",
            f"base>={base_reference.get('threshold', 0.0):.2f}",
            str(base_reference.get("rows", 0)),
            format_compact_metric_value((base_reference.get("win_rate") or 0.0) * 100, percent=True, decimals=2),
            str(base_reference.get("wins", 0)),
            str(base_reference.get("losses", 0)),
            format_compact_metric_value(base_reference.get("net_pnl_pct"), decimals=3),
            format_compact_metric_value(base_reference.get("avg_pnl_pct"), decimals=4),
        )
    ]

    if best_default_by_wins:
        rows.append(
            (
                "Best wins",
                f"base>={best_default_by_wins['base_threshold']:.2f}, meta>={best_default_by_wins['meta_threshold']:.3f}",
                str(best_default_by_wins["rows"]),
                format_compact_metric_value((best_default_by_wins.get("win_rate") or 0.0) * 100, percent=True, decimals=2),
                str(best_default_by_wins["wins"]),
                str(best_default_by_wins["losses"]),
                format_compact_metric_value(best_default_by_wins.get("net_pnl_pct"), decimals=3),
                format_compact_metric_value(best_default_by_wins.get("avg_pnl_pct"), decimals=4),
            )
        )

    if best_default_by_net_pnl:
        label = "Best net pnl"
        if best_default_by_wins and (
            best_default_by_net_pnl["base_threshold"] == best_default_by_wins["base_threshold"]
            and best_default_by_net_pnl["meta_threshold"] == best_default_by_wins["meta_threshold"]
        ):
            label = "Best net pnl*"
        rows.append(
            (
                label,
                f"base>={best_default_by_net_pnl['base_threshold']:.2f}, meta>={best_default_by_net_pnl['meta_threshold']:.3f}",
                str(best_default_by_net_pnl["rows"]),
                format_compact_metric_value((best_default_by_net_pnl.get("win_rate") or 0.0) * 100, percent=True, decimals=2),
                str(best_default_by_net_pnl["wins"]),
                str(best_default_by_net_pnl["losses"]),
                format_compact_metric_value(best_default_by_net_pnl.get("net_pnl_pct"), decimals=3),
                format_compact_metric_value(best_default_by_net_pnl.get("avg_pnl_pct"), decimals=4),
            )
        )

    if best_default_by_win_rate:
        label = "Best win rate"
        if best_default_by_wins and (
            best_default_by_win_rate["base_threshold"] == best_default_by_wins["base_threshold"]
            and best_default_by_win_rate["meta_threshold"] == best_default_by_wins["meta_threshold"]
        ):
            label = "Best win rate*"
        rows.append(
            (
                label,
                f"base>={best_default_by_win_rate['base_threshold']:.2f}, meta>={best_default_by_win_rate['meta_threshold']:.3f}",
                str(best_default_by_win_rate["rows"]),
                format_compact_metric_value((best_default_by_win_rate.get("win_rate") or 0.0) * 100, percent=True, decimals=2),
                str(best_default_by_win_rate["wins"]),
                str(best_default_by_win_rate["losses"]),
                format_compact_metric_value(best_default_by_win_rate.get("net_pnl_pct"), decimals=3),
                format_compact_metric_value(best_default_by_win_rate.get("avg_pnl_pct"), decimals=4),
            )
        )

    headers = ["Mode", "Thresholds", "Signals", "Win rate", "Wins", "Losses", "Net pnl", "Avg pnl"]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def render_border():
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render_row(values):
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        render_border(),
        render_row(headers),
        render_border(),
    ]
    for row in rows:
        lines.append(render_row(list(row)))
    lines.append(render_border())
    lines.append(
        f"Meta OOS rows: {meta_payload.get('rows_with_meta_oos', 0)} | folds: {', '.join(map(str, meta_payload.get('folds_with_meta_oos', [])))} | cost={meta_payload.get('roundtrip_cost_pct', 0.0):.4f}"
    )
    if best_default_by_wins:
        lines.append(
            f"Delta vs base: wins {best_default_by_wins['delta_wins_vs_base']:+d}, losses {best_default_by_wins['delta_losses_vs_base']:+d}, net pnl {best_default_by_wins.get('delta_net_pnl_pct_vs_base', 0.0):+.3f}"
        )
    return lines


def log_meta_filter_summary(meta_payload):
    logger.info("=" * 72)
    logger.info("Meta-filter comparison (base vs base+take/skip):")
    for line in build_meta_filter_summary_lines(meta_payload):
        logger.info(line)


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

        # ── Step 1: Walk-Forward Validation → honest OOS metrics ──────────
        oos_metrics, fold_details, median_best_iter, oos_prediction_frame = walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
        )
        meta_filter_payload = evaluate_meta_filter_comparison(
            oos_prediction_frame=oos_prediction_frame,
            feature_columns=feature_columns,
            seed=args.seed,
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
            "fold_stability": build_fold_stability_payload(fold_details),
            "median_best_iteration": median_best_iter,
            "total_rows": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "n_splits": args.n_splits,
            "purge_gap": args.purge_gap,
            "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            "candidate_rows": int(dataset.attrs.get("candidate_rows", len(dataset))),
            "excluded_by_event_filter_rows": int(dataset.attrs.get("excluded_by_event_filter_rows", 0)),
            "event_filter": dataset.attrs.get("event_filter_config"),
            "experiment": experiment_snapshot,
            "dataset_period": build_period_payload(dataset),
            "meta_filter": meta_filter_payload,
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
        if meta_filter_payload:
            log_meta_filter_summary(meta_filter_payload)

        log_feature_importance_ranking(prod_model, feature_columns)
        save_directional_artifacts(prod_model, metrics, dataset, feature_columns, prod_clip_bounds, args)
        history_path = get_train_history_path(args.model_name)
        history_entry = build_train_history_entry(args, metrics, experiment_snapshot)
        history = save_train_history(history_path, history_entry)
        logger.info("Saved train history to %s", history_path)
        log_train_history_summary(history_entry, history, limit=10)

    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
