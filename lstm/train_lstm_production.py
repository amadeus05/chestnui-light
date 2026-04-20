import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as cfg
from lstm import config_lstm as lstm_cfg
from lstm.dataset import SequenceDataset, SequenceStandardizer
from lstm.model import LSTMClassifier
from lstm.train_lstm_walk_forward import (
    load_frames,
    run_epoch,
    set_seed,
    split_train_eval_indices,
)
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one production LSTM on all available directional samples.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument("--symbols", nargs="+", default=cfg.SYMBOLS, help="Symbols to load.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=lstm_cfg.SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=lstm_cfg.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=lstm_cfg.EPOCHS)
    parser.add_argument("--hidden-size", type=int, default=lstm_cfg.HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=lstm_cfg.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=lstm_cfg.DROPOUT)
    parser.add_argument("--lr", type=float, default=lstm_cfg.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=lstm_cfg.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=lstm_cfg.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--model-name", default="lstm_target_production")
    return parser.parse_args()


def train_production_model(dataset: SequenceDataset, train_indices: list[int], eval_indices: list[int], args, device):
    fit_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        Subset(dataset, eval_indices or train_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = LSTMClassifier(
        input_size=len(dataset.feature_columns),
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
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        logger.info("epoch=%s train_loss=%.5f eval_loss=%.5f", epoch, train_loss, eval_loss)
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_eval_loss


def save_production_payload(model, standardizer, feature_columns, dataset, best_epoch, best_eval_loss, args):
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    model_path = cfg.MODELS_DIR / f"{args.model_name}.pt"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"

    payload = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "standardizer": standardizer.to_payload(),
        "sequence_length": int(args.sequence_length),
        "symbols": list(args.symbols),
        "model_args": {
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "input_size": int(len(feature_columns)),
        },
        "training": {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rows": int(len(dataset)),
            "best_epoch": int(best_epoch),
            "best_eval_loss": float(best_eval_loss),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
        },
    }
    torch.save(payload, model_path)
    features_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "sequence_length": int(args.sequence_length),
                "symbols": list(args.symbols),
                "model_args": payload["model_args"],
                "training": payload["training"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved production LSTM model to %s", model_path)
    logger.info("Saved production LSTM feature metadata to %s", features_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    sample_frame, _, history_by_symbol, feature_columns = load_frames(args.db_path, args.symbols)
    raw_dataset = SequenceDataset(
        sample_frame,
        history_by_symbol,
        feature_columns,
        args.sequence_length,
    )
    if len(raw_dataset) < lstm_cfg.MIN_TRAIN_ROWS:
        raise RuntimeError(f"Only {len(raw_dataset)} sequences, need at least {lstm_cfg.MIN_TRAIN_ROWS}.")

    train_indices, eval_indices = split_train_eval_indices(
        len(raw_dataset),
        lstm_cfg.VALIDATION_FRACTION,
    )
    standardizer = SequenceStandardizer().fit(raw_dataset.sequences_array(train_indices))
    dataset = SequenceDataset(
        sample_frame,
        history_by_symbol,
        feature_columns,
        args.sequence_length,
        standardizer=standardizer,
    )

    logger.info(
        "Training production LSTM | rows=%s | fit=%s | eval=%s | features=%s | seq_len=%s",
        len(dataset),
        len(train_indices),
        len(eval_indices),
        len(feature_columns),
        args.sequence_length,
    )
    model, best_epoch, best_eval_loss = train_production_model(
        dataset,
        train_indices,
        eval_indices,
        args,
        device,
    )
    save_production_payload(model, standardizer, feature_columns, dataset, best_epoch, best_eval_loss, args)


if __name__ == "__main__":
    main()
