"""Walk-forward OOS-предикты + сохранение артефактов + вызов портфельного бэктеста."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config as cfg
import train
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.application.backtest_engine import backtest
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


def apply_end_date_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return frame

    end_cutoff = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_cutoff):
        return frame

    return frame.loc[frame[train.TIMESTAMP_COLUMN] <= end_cutoff].copy()


def load_candidate_and_training_frames(db_path: str, symbols: list[str]):
    repository = HistoricalKlineRepository(db_path=db_path)
    frame = repository.load_feature_dataset(symbols)
    frame = frame.dropna(subset=[train.TIMESTAMP_COLUMN, train.TARGET_COLUMN])
    frame = frame.sort_values(train.TIMESTAMP_COLUMN).reset_index(drop=True)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame = apply_end_date_cutoff(frame)

    event_filter_config = resolve_event_filter_config()
    candidate_mask = build_candidate_event_mask(frame, event_filter_config)
    candidate_frame = frame.loc[candidate_mask].copy()

    directional_frame = candidate_frame.loc[candidate_frame[train.TARGET_COLUMN].astype(int) != 0].copy()
    directional_frame[train.TARGET_COLUMN] = directional_frame[train.TARGET_COLUMN].astype(int).map(train.LABEL_TO_CLASS)

    symbol_categories = list(dict.fromkeys(symbols))
    candidate_frame[train.SYMBOL_COLUMN] = pd.Categorical(
        candidate_frame[train.SYMBOL_COLUMN],
        categories=symbol_categories,
    )
    directional_frame[train.SYMBOL_COLUMN] = pd.Categorical(
        directional_frame[train.SYMBOL_COLUMN],
        categories=symbol_categories,
    )

    feature_columns = train.select_feature_columns(directional_frame)
    return frame, candidate_frame, directional_frame, feature_columns, event_filter_config


def fit_fold_model(train_df: pd.DataFrame, feature_columns: list[str], seed: int):
    clip_bounds = train.build_feature_clip_bounds(train_df, feature_columns)
    clipped_train = train.apply_feature_clip_bounds(train_df, clip_bounds)

    x_train = clipped_train[feature_columns]
    y_train = clipped_train[train.TARGET_COLUMN]
    w_train = train.compute_sample_weights(clipped_train[train.TIMESTAMP_COLUMN])

    model = train.build_model(seed=seed)
    categorical_feature = [train.SYMBOL_COLUMN] if train.SYMBOL_COLUMN in feature_columns else "auto"

    if len(x_train) >= 20:
        internal_eval_size = max(1, int(len(x_train) * 0.15))
        if internal_eval_size < len(x_train):
            model.fit(
                x_train.iloc[:-internal_eval_size],
                y_train.iloc[:-internal_eval_size],
                sample_weight=w_train[:-internal_eval_size],
                eval_set=[(x_train.iloc[-internal_eval_size:], y_train.iloc[-internal_eval_size:])],
                eval_metric="binary_logloss",
                categorical_feature=categorical_feature,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            return model, clip_bounds

    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        categorical_feature=categorical_feature,
    )
    return model, clip_bounds


def build_walk_forward_predictions(
    full_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    directional_frame: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    unique_ts = np.sort(full_frame[train.TIMESTAMP_COLUMN].dropna().unique())
    if len(unique_ts) < n_splits + 1:
        raise RuntimeError(
            f"Only {len(unique_ts)} unique timestamps, need at least {n_splits + 1} for {n_splits} folds."
        )

    predictions = []
    fold_details = []
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(splitter.split(unique_ts), start=1):
        train_timestamps = unique_ts[train_ts_idx]
        test_timestamps = unique_ts[test_ts_idx]

        if purge_gap > 0 and len(train_timestamps) > purge_gap:
            train_timestamps = train_timestamps[:-purge_gap]

        train_df = directional_frame.loc[
            directional_frame[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))
        ].copy()
        test_df = candidate_frame.loc[
            candidate_frame[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))
        ].copy()

        train_classes = sorted(train_df[train.TARGET_COLUMN].dropna().unique().tolist())
        if train_df.empty or test_df.empty or len(train_classes) < 2:
            print(f"Fold {fold_idx}: skipped (train={len(train_df)}, test={len(test_df)}, classes={train_classes})")
            continue

        model, clip_bounds = fit_fold_model(train_df, feature_columns, seed + fold_idx)
        clipped_test = train.apply_feature_clip_bounds(test_df, clip_bounds)
        clipped_test = clipped_test.dropna(subset=feature_columns)
        if clipped_test.empty:
            print(f"Fold {fold_idx}: skipped after feature NaN cleanup")
            continue

        proba = model.predict_proba(clipped_test[feature_columns])
        fold_predictions = pd.DataFrame(
            {
                "timestamp": clipped_test[train.TIMESTAMP_COLUMN].values,
                "symbol": clipped_test[train.SYMBOL_COLUMN].astype(str).values,
                "p_short": proba[:, 0],
                "p_long": proba[:, 1],
                "fold": fold_idx,
            }
        )
        predictions.append(fold_predictions)

        fold_info = {
            "fold": fold_idx,
            "train_rows": int(len(train_df)),
            "prediction_rows": int(len(fold_predictions)),
            "purged_timestamps": int(purge_gap),
            "train_start": str(train_df[train.TIMESTAMP_COLUMN].min()),
            "train_end": str(train_df[train.TIMESTAMP_COLUMN].max()),
            "test_start": str(clipped_test[train.TIMESTAMP_COLUMN].min()),
            "test_end": str(clipped_test[train.TIMESTAMP_COLUMN].max()),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or getattr(model, "n_estimators_", 0)),
        }
        fold_details.append(fold_info)
        print(
            f"Fold {fold_idx}/{n_splits}: train={fold_info['train_rows']} "
            f"pred={fold_info['prediction_rows']} "
            f"[{fold_info['test_start']} -> {fold_info['test_end']}]"
        )

    if not predictions:
        raise RuntimeError("All walk-forward folds were skipped.")

    return pd.concat(predictions, ignore_index=True), fold_details


def iter_monthly_splits(unique_ts, train_months: int, test_months: int, window_mode: str = "expanding"):
    timestamps = pd.Series(pd.to_datetime(unique_ts)).dropna().sort_values()
    if timestamps.empty:
        return
    if window_mode not in {"expanding", "rolling"}:
        raise ValueError("window_mode must be either 'expanding' or 'rolling'.")

    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    train_end = first_ts + pd.DateOffset(months=train_months)
    fold_idx = 1

    while train_end < last_ts:
        test_end = train_end + pd.DateOffset(months=test_months)
        if window_mode == "rolling":
            train_start = train_end - pd.DateOffset(months=train_months)
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
        else:
            train_mask = timestamps < train_end
        test_mask = (timestamps >= train_end) & (timestamps < test_end)

        train_timestamps = timestamps.loc[train_mask].to_numpy()
        test_timestamps = timestamps.loc[test_mask].to_numpy()
        if len(train_timestamps) > 0 and len(test_timestamps) > 0:
            yield fold_idx, train_timestamps, test_timestamps
            fold_idx += 1

        train_end = test_end


def build_monthly_walk_forward_predictions(
    full_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    directional_frame: pd.DataFrame,
    feature_columns: list[str],
    train_months: int,
    test_months: int,
    window_mode: str,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if train_months <= 0 or test_months <= 0:
        raise ValueError("monthly train/test windows must be positive month counts.")

    unique_ts = np.sort(full_frame[train.TIMESTAMP_COLUMN].dropna().unique())
    predictions = []
    fold_details = []
    splits = list(iter_monthly_splits(unique_ts, train_months, test_months, window_mode=window_mode))

    for fold_idx, train_timestamps, test_timestamps in splits:
        original_train_timestamps = train_timestamps
        if purge_gap > 0 and len(train_timestamps) > purge_gap:
            train_timestamps = train_timestamps[:-purge_gap]

        train_df = directional_frame.loc[
            directional_frame[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))
        ].copy()
        test_df = candidate_frame.loc[
            candidate_frame[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))
        ].copy()

        train_classes = sorted(train_df[train.TARGET_COLUMN].dropna().unique().tolist())
        if train_df.empty or test_df.empty or len(train_classes) < 2:
            print(f"Fold {fold_idx}: skipped (train={len(train_df)}, test={len(test_df)}, classes={train_classes})")
            continue

        model, clip_bounds = fit_fold_model(train_df, feature_columns, seed + fold_idx)
        clipped_test = train.apply_feature_clip_bounds(test_df, clip_bounds)
        clipped_test = clipped_test.dropna(subset=feature_columns)
        if clipped_test.empty:
            print(f"Fold {fold_idx}: skipped after feature NaN cleanup")
            continue

        proba = model.predict_proba(clipped_test[feature_columns])
        fold_predictions = pd.DataFrame(
            {
                "timestamp": clipped_test[train.TIMESTAMP_COLUMN].values,
                "symbol": clipped_test[train.SYMBOL_COLUMN].astype(str).values,
                "p_short": proba[:, 0],
                "p_long": proba[:, 1],
                "fold": fold_idx,
            }
        )
        predictions.append(fold_predictions)

        fold_info = {
            "fold": fold_idx,
            "split_mode": "monthly",
            "monthly_window_mode": window_mode,
            "train_rows": int(len(train_df)),
            "prediction_rows": int(len(fold_predictions)),
            "purged_timestamps": int(purge_gap),
            "train_start": str(train_df[train.TIMESTAMP_COLUMN].min()),
            "train_end": str(train_df[train.TIMESTAMP_COLUMN].max()),
            "train_end_before_purge": str(pd.to_datetime(original_train_timestamps[-1])),
            "test_start": str(clipped_test[train.TIMESTAMP_COLUMN].min()),
            "test_end": str(clipped_test[train.TIMESTAMP_COLUMN].max()),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or getattr(model, "n_estimators_", 0)),
        }
        fold_details.append(fold_info)
        print(
            f"Fold {fold_idx}/{len(splits)} monthly: train={fold_info['train_rows']} "
            f"pred={fold_info['prediction_rows']} "
            f"[{fold_info['test_start']} -> {fold_info['test_end']}]"
        )

    if not predictions:
        raise RuntimeError("All monthly walk-forward folds were skipped.")

    return pd.concat(predictions, ignore_index=True), fold_details


def build_features_meta(
    predictions: pd.DataFrame,
    feature_columns: list[str],
    symbols: list[str],
    event_filter_config: dict,
    args: argparse.Namespace,
):
    return {
        "feature_columns": feature_columns,
        "label_mapping": {"short": 0, "long": 1},
        "inverse_label_mapping": {str(key): value for key, value in train.CLASS_TO_LABEL.items()},
        "symbols": list(symbols),
        "rows": int(len(predictions)),
        "task_type": "binary_directional_walk_forward_oos",
        "train_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "wfv_n_splits": int(args.n_splits),
        "wfv_purge_gap": int(args.purge_gap),
        "wfv_split_mode": str(args.split_mode),
        "wfv_monthly_train_months": int(args.monthly_train_months),
        "wfv_monthly_test_months": int(args.monthly_test_months),
        "wfv_monthly_window_mode": str(args.monthly_window_mode),
        "event_filter": event_filter_config,
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": {},
            "note": "Fold-specific clipping was applied before creating OOS predictions.",
        },
    }


def save_walk_forward_payload(predictions: pd.DataFrame, fold_details: list[dict], args: argparse.Namespace) -> None:
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    predictions_path = cfg.MODELS_DIR / f"{args.predictions_name}.csv"
    summary_path = cfg.MODELS_DIR / f"{args.predictions_name}_summary.json"

    predictions.to_csv(predictions_path, index=False)
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbols": list(args.symbols),
        "n_splits": int(args.n_splits),
        "purge_gap": int(args.purge_gap),
        "split_mode": str(args.split_mode),
        "monthly_train_months": int(args.monthly_train_months),
        "monthly_test_months": int(args.monthly_test_months),
        "monthly_window_mode": str(args.monthly_window_mode),
        "prediction_rows": int(len(predictions)),
        "prediction_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "fold_details": fold_details,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved OOS predictions: {predictions_path}")
    print(f"Saved WFV summary: {summary_path}")


def run_walk_forward_oos(args: argparse.Namespace) -> None:
    full_frame, candidate_frame, directional_frame, feature_columns, event_filter_config = load_candidate_and_training_frames(
        args.db_path,
        list(args.symbols),
    )
    print(
        f"Loaded candidate rows={len(candidate_frame)}, directional train rows={len(directional_frame)}, "
        f"features={len(feature_columns)}"
    )

    if args.split_mode == "monthly":
        predictions, fold_details = build_monthly_walk_forward_predictions(
            full_frame=full_frame,
            candidate_frame=candidate_frame,
            directional_frame=directional_frame,
            feature_columns=feature_columns,
            train_months=args.monthly_train_months,
            test_months=args.monthly_test_months,
            window_mode=args.monthly_window_mode,
            purge_gap=args.purge_gap,
            seed=args.seed,
        )
    else:
        predictions, fold_details = build_walk_forward_predictions(
            full_frame=full_frame,
            candidate_frame=candidate_frame,
            directional_frame=directional_frame,
            feature_columns=feature_columns,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
            seed=args.seed,
        )
    save_walk_forward_payload(predictions, fold_details, args)

    features_meta = build_features_meta(
        predictions=predictions,
        feature_columns=feature_columns,
        symbols=list(args.symbols),
        event_filter_config=event_filter_config,
        args=args,
    )
    chart_path = cfg.BACKTEST_CHARTS_DIR / "equity_curve_walk_forward.png"
    backtest(
        features_meta=features_meta,
        predictions=predictions,
        equity_curve_path=chart_path,
        result_title="WALK-FORWARD OOS BACKTEST",
    )
