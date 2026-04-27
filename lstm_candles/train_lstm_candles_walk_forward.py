import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bt
import config as cfg
import train
from lstm.dataset import SequenceDataset, SequenceStandardizer, build_history_by_symbol
from lstm.model import LSTMClassifier
from lstm_candles import config_lstm_candles as raw_cfg
from lstm_candles.data import load_candle_sequence_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM on candle-sequence semi-raw inputs with walk-forward OOS.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument("--symbols", nargs="+", default=cfg.SYMBOLS, help="Symbols to load.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--split-mode", choices=["tscv", "monthly"], default="monthly")
    parser.add_argument("--monthly-train-months", type=int, default=6)
    parser.add_argument("--monthly-test-months", type=int, default=1)
    parser.add_argument("--monthly-window-mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--purge-gap", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=raw_cfg.SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=raw_cfg.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=raw_cfg.EPOCHS)
    parser.add_argument("--hidden-size", type=int, default=raw_cfg.HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=raw_cfg.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=raw_cfg.DROPOUT)
    parser.add_argument("--lr", type=float, default=raw_cfg.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=raw_cfg.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=raw_cfg.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=raw_cfg.MIN_EPOCHS_BEFORE_EARLY_STOP)
    parser.add_argument("--early-stopping-min-delta", type=float, default=raw_cfg.EARLY_STOPPING_MIN_DELTA)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--model-name", default=raw_cfg.MODEL_NAME)
    parser.add_argument("--predictions-name", default=raw_cfg.PREDICTIONS_NAME)
    parser.add_argument("--skip-backtest", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_eval_indices(n_items: int, validation_fraction: float):
    if n_items < 2:
        return list(range(n_items)), []
    eval_size = max(1, int(n_items * validation_fraction))
    if eval_size >= n_items:
        eval_size = 1
    split_at = n_items - eval_size
    return list(range(split_at)), list(range(split_at, n_items))


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    model.train(train_mode)
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), raw_cfg.GRADIENT_CLIP)
                optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / max(total_rows, 1)


def predict_dataset(model, dataset: SequenceDataset, batch_size: int, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probas = []
    preds = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            probas.append(proba)
            preds.append(np.argmax(proba, axis=1))
    if not probas:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(probas), np.concatenate(preds)


def train_fold_model(train_dataset: SequenceDataset, args, device, train_indices: list[int], eval_indices: list[int]):
    fit_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(train_dataset, eval_indices or train_indices), batch_size=args.batch_size, shuffle=False)

    model = LSTMClassifier(
        input_size=len(train_dataset.feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_eval_loss = float("inf")
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, fit_loader, criterion, optimizer, device, train_mode=True)
        eval_loss = run_epoch(model, eval_loader, criterion, optimizer, device, train_mode=False)
        if eval_loss < (best_eval_loss - args.early_stopping_min_delta):
            best_eval_loss = eval_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        logger.info("epoch=%s train_loss=%.5f eval_loss=%.5f", epoch, train_loss, eval_loss)
        if epoch >= args.min_epochs_before_early_stop and stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_eval_loss


def load_frames(db_path: str, symbols: list[str]):
    sample_frame = train.load_training_frame(db_path, symbols)
    candle_frame, feature_columns = load_candle_sequence_frames(db_path, symbols)
    history_by_symbol = build_history_by_symbol(candle_frame, feature_columns)
    return sample_frame, history_by_symbol, feature_columns


def build_features_meta(predictions: pd.DataFrame, feature_columns: list[str], symbols: list[str], args) -> dict:
    return {
        "feature_columns": feature_columns,
        "label_mapping": {"short": 0, "long": 1},
        "inverse_label_mapping": {str(key): value for key, value in train.CLASS_TO_LABEL.items()},
        "symbols": list(symbols),
        "rows": int(len(predictions)),
        "task_type": "lstm_candles_binary_directional_walk_forward_oos",
        "train_period": {
            "start": str(pd.to_datetime(predictions[train.TIMESTAMP_COLUMN]).min()),
            "end": str(pd.to_datetime(predictions[train.TIMESTAMP_COLUMN]).max()),
        },
        "wfv_n_splits": int(args.n_splits),
        "wfv_purge_gap": int(args.purge_gap),
        "wfv_split_mode": str(args.split_mode),
        "wfv_monthly_train_months": int(args.monthly_train_months),
        "wfv_monthly_test_months": int(args.monthly_test_months),
        "wfv_monthly_window_mode": str(args.monthly_window_mode),
        "event_filter": train.resolve_event_filter_config() if hasattr(train, "resolve_event_filter_config") else {},
        "feature_clip": {
            "enabled": False,
            "bounds": {},
            "note": "Candle-sequence LSTM uses fold-local standardization.",
        },
    }


def walk_forward_lstm(sample_frame, history_by_symbol, feature_columns, args, device):
    unique_ts = np.asarray(sample_frame.attrs.get("all_timestamps", np.sort(sample_frame[train.TIMESTAMP_COLUMN].unique())))
    timestamp_splits = train.build_timestamp_splits(
        unique_ts=unique_ts,
        n_splits=args.n_splits,
        split_mode=args.split_mode,
        monthly_train_months=args.monthly_train_months,
        monthly_test_months=args.monthly_test_months,
        monthly_window_mode=args.monthly_window_mode,
    )
    if args.max_folds is not None:
        timestamp_splits = timestamp_splits[: args.max_folds]

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    prediction_frames = []
    fold_details = []
    last_model_state = None

    for fold_idx, original_train_timestamps, test_timestamps in timestamp_splits:
        train_timestamps = original_train_timestamps
        if args.purge_gap > 0 and len(train_timestamps) > args.purge_gap:
            train_timestamps = train_timestamps[:-args.purge_gap]

        train_df = sample_frame.loc[sample_frame[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))].copy()
        test_df = sample_frame.loc[sample_frame[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))].copy()
        if train_df.empty or test_df.empty:
            logger.warning("Fold %s skipped: empty train/test.", fold_idx)
            continue
        if len(sorted(train_df[train.TARGET_COLUMN].unique().tolist())) < 2:
            logger.warning("Fold %s skipped: train has one class.", fold_idx)
            continue

        raw_train_dataset = SequenceDataset(train_df, history_by_symbol, feature_columns, args.sequence_length)
        if len(raw_train_dataset) < raw_cfg.MIN_TRAIN_ROWS:
            logger.warning("Fold %s skipped: only %s train sequences.", fold_idx, len(raw_train_dataset))
            continue

        train_indices, eval_indices = split_train_eval_indices(len(raw_train_dataset), raw_cfg.VALIDATION_FRACTION)
        standardizer = SequenceStandardizer().fit(raw_train_dataset.sequences_array(train_indices))
        train_dataset = SequenceDataset(train_df, history_by_symbol, feature_columns, args.sequence_length, standardizer=standardizer)
        test_dataset = SequenceDataset(test_df, history_by_symbol, feature_columns, args.sequence_length, standardizer=standardizer)
        if len(test_dataset) == 0:
            logger.warning("Fold %s skipped: no test sequences.", fold_idx)
            continue

        logger.info(
            "Fold %s/%s | train_seq=%s | test_seq=%s | features=%s",
            fold_idx,
            len(timestamp_splits),
            len(train_dataset),
            len(test_dataset),
            len(feature_columns),
        )
        model, best_epoch, best_eval_loss = train_fold_model(
            train_dataset,
            args,
            device,
            train_indices=train_indices,
            eval_indices=eval_indices,
        )
        proba, pred = predict_dataset(model, test_dataset, args.batch_size, device)
        meta = test_dataset.metadata_frame()
        y_true = meta[train.TARGET_COLUMN].to_numpy(dtype=int)

        all_y_true.append(y_true)
        all_y_pred.append(pred)
        all_y_proba.append(proba)
        prediction_frames.append(
            pd.DataFrame(
                {
                    train.TIMESTAMP_COLUMN: meta[train.TIMESTAMP_COLUMN].values,
                    train.SYMBOL_COLUMN: meta[train.SYMBOL_COLUMN].values,
                    train.TARGET_COLUMN: y_true,
                    "prediction": pred,
                    "p_short": proba[:, 0],
                    "p_long": proba[:, 1],
                    "fold": fold_idx,
                }
            )
        )
        fold_details.append(
            {
                "fold": int(fold_idx),
                "split_mode": args.split_mode,
                "train_sequences": int(len(train_dataset)),
                "test_sequences": int(len(test_dataset)),
                "best_epoch": int(best_epoch),
                "best_eval_loss": float(best_eval_loss),
                "accuracy": float(accuracy_score(y_true, pred)),
                "roc_auc": float(roc_auc_score(y_true, proba[:, 1])) if len(set(y_true)) > 1 else None,
                "timestamp_boundaries": {
                    "train_start": train.format_timestamp(train_timestamps[0]),
                    "train_end_after_purge": train.format_timestamp(train_timestamps[-1]),
                    "test_start": train.format_timestamp(test_timestamps[0]),
                    "test_end": train.format_timestamp(test_timestamps[-1]),
                },
            }
        )
        last_model_state = {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "standardizer": standardizer.to_payload(),
            "args": vars(args),
        }

    if not prediction_frames:
        raise RuntimeError("All candle-sequence LSTM walk-forward folds were skipped.")

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    y_proba_all = np.vstack(all_y_proba)
    metrics = train.evaluate_model(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_proba=y_proba_all,
        split_name="oos",
        n_rows=len(y_true_all),
    )
    return metrics, fold_details, pd.concat(prediction_frames, ignore_index=True), last_model_state


def save_payload(metrics, fold_details, predictions, model_state, feature_columns, args):
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    predictions_path = cfg.MODELS_DIR / f"{args.predictions_name}.csv"
    metrics_path = cfg.MODELS_DIR / f"{args.model_name}_metrics.json"
    model_path = cfg.MODELS_DIR / f"{args.model_name}.pt"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"

    predictions.to_csv(predictions_path, index=False)
    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_type": "LSTMClassifier",
        "symbols": list(args.symbols),
        "sequence_length": int(args.sequence_length),
        "feature_count": int(len(feature_columns)),
        "oos_metrics": metrics,
        "fold_details": fold_details,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    features_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "sequence_length": int(args.sequence_length),
                "symbols": list(args.symbols),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if model_state is not None:
        torch.save(model_state, model_path)

    logger.info("Saved candle-LSTM predictions to %s", predictions_path)
    logger.info("Saved candle-LSTM metrics to %s", metrics_path)
    logger.info("Saved candle-LSTM model snapshot to %s", model_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    sample_frame, history_by_symbol, feature_columns = load_frames(args.db_path, args.symbols)
    logger.info(
        "Loaded candle-LSTM sample rows=%s | symbols=%s | features=%s | seq_len=%s",
        len(sample_frame),
        ", ".join(args.symbols),
        len(feature_columns),
        args.sequence_length,
    )

    metrics, fold_details, predictions, model_state = walk_forward_lstm(
        sample_frame=sample_frame,
        history_by_symbol=history_by_symbol,
        feature_columns=feature_columns,
        args=args,
        device=device,
    )
    save_payload(metrics, fold_details, predictions, model_state, feature_columns, args)
    logger.info(
        "Candle-LSTM OOS | acc=%.4f | bal_acc=%.4f | f1=%.4f | auc=%.4f | mcc=%.4f",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
        metrics["roc_auc"],
        metrics["mcc"],
    )

    if not args.skip_backtest:
        features_meta = build_features_meta(predictions, feature_columns, args.symbols, args)
        chart_path = cfg.BACKTEST_CHARTS_DIR / "equity_curve_lstm_candles_walk_forward.png"
        bt.backtest(
            features_meta=features_meta,
            predictions=predictions,
            equity_curve_path=chart_path,
            result_title="LSTM CANDLES WALK-FORWARD OOS BACKTEST",
        )


if __name__ == "__main__":
    main()

