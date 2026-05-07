from __future__ import annotations

import argparse
from typing import Sequence

import config as cfg
from lstm import config_lstm as lstm_cfg
from src.models.base import BaseModelRunner
from src.models.lstm.artifacts import LstmArtifactWriter
from src.models.lstm.trainer import LstmTrainer, LstmTrainRequest


class LstmRunner(BaseModelRunner):
    """Runner for the new LSTM production path plus legacy WFV/replay."""

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_new_production_train(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_train_args(argv)
        trained = LstmTrainer().train_production(
            LstmTrainRequest(
                db_path=args.db_path,
                symbols=args.symbols,
                seed=args.seed,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                min_epochs_before_early_stop=args.min_epochs_before_early_stop,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        )
        paths = LstmArtifactWriter(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).save_production_model(trained)
        print(f"Saved '{self.spec.key}' production artifacts to {paths.root}")

    @staticmethod
    def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train production LSTM model via the new model runner.")
        parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
        parser.add_argument("--symbols", nargs="+", default=list(cfg.SYMBOLS), help="Symbols to load.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--sequence-length", type=int, default=lstm_cfg.SEQUENCE_LENGTH)
        parser.add_argument("--batch-size", type=int, default=lstm_cfg.BATCH_SIZE)
        parser.add_argument("--epochs", type=int, default=lstm_cfg.EPOCHS)
        parser.add_argument("--hidden-size", type=int, default=lstm_cfg.HIDDEN_SIZE)
        parser.add_argument("--num-layers", type=int, default=lstm_cfg.NUM_LAYERS)
        parser.add_argument("--dropout", type=float, default=lstm_cfg.DROPOUT)
        parser.add_argument("--lr", type=float, default=lstm_cfg.LEARNING_RATE)
        parser.add_argument("--weight-decay", type=float, default=lstm_cfg.WEIGHT_DECAY)
        parser.add_argument("--patience", type=int, default=lstm_cfg.EARLY_STOPPING_PATIENCE)
        parser.add_argument("--min-epochs-before-early-stop", type=int, default=lstm_cfg.MIN_EPOCHS_BEFORE_EARLY_STOP)
        parser.add_argument("--early-stopping-min-delta", type=float, default=lstm_cfg.EARLY_STOPPING_MIN_DELTA)
        return parser.parse_args(list(argv or []))
