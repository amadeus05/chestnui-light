from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import config as cfg
from lstm import config_lstm as lstm_cfg
from src.models.base import BaseModelRunner
from src.models.backtest import PredictionBacktestRequest, PredictionBacktestRunner
from src.models.lstm.artifacts import LstmArtifactWriter
from src.models.lstm.trainer import LstmTrainer, LstmTrainRequest
from src.models.lstm.walk_forward import LstmWalkForwardRequest, LstmWalkForwardRunner


class LstmRunner(BaseModelRunner):
    """Runner for the new LSTM production path plus legacy WFV/replay."""

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_walk_forward(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_walk_forward_args(argv)
        request = LstmWalkForwardRequest(
            db_path=args.db_path,
            symbols=args.symbols,
            seed=args.seed,
            n_splits=args.n_splits,
            split_mode=args.split_mode,
            monthly_train_months=args.monthly_train_months,
            monthly_test_months=args.monthly_test_months,
            monthly_window_mode=args.monthly_window_mode,
            purge_gap=args.purge_gap,
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
            max_folds=args.max_folds,
        )
        if args.dry_run:
            plan = LstmWalkForwardRunner().plan(request)
            self.print_walk_forward_plan(plan)
            return

        result = LstmWalkForwardRunner().run(request)
        paths = LstmArtifactWriter(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).save_walk_forward_result(result)
        print(f"Saved '{self.spec.key}' walk-forward artifacts to {paths.root}")
        self.print_walk_forward_summary(result)
        if not args.skip_backtest:
            PredictionBacktestRunner(
                spec=self.spec,
                artifact_store=self.artifact_store,
            ).run(
                PredictionBacktestRequest(
                    predictions_path=paths.predictions,
                    metadata_path=paths.metadata,
                    chart_path=cfg.BACKTEST_CHARTS_DIR / self.spec.default_chart_name,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
            )

    def run_backtest(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_backtest_args(argv)
        PredictionBacktestRunner(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).run(
            PredictionBacktestRequest(
                predictions_path=args.predictions,
                metadata_path=args.metadata,
                chart_path=args.chart,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        )

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

    @staticmethod
    def parse_walk_forward_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Run LSTM walk-forward OOS via the new model runner.")
        parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
        parser.add_argument("--symbols", nargs="+", default=list(cfg.SYMBOLS), help="Symbols to load.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--n-splits", type=int, default=5)
        parser.add_argument("--split-mode", choices=["tscv", "monthly"], default="tscv")
        parser.add_argument("--monthly-train-months", type=int, default=6)
        parser.add_argument("--monthly-test-months", type=int, default=1)
        parser.add_argument("--monthly-window-mode", choices=["expanding", "rolling"], default="expanding")
        parser.add_argument("--purge-gap", type=int, default=cfg.effective_max_label_horizon())
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
        parser.add_argument("--max-folds", type=int, default=None)
        parser.add_argument("--skip-backtest", action="store_true", help="Only build and save OOS predictions.")
        parser.add_argument("--backtest-start-date", default=None, help="Optional replay backtest start timestamp.")
        parser.add_argument("--backtest-end-date", default=None, help="Optional replay backtest end timestamp.")
        parser.add_argument("--dry-run", action="store_true", help="Print the WFV plan without training folds.")
        return parser.parse_args(list(argv or []))

    @staticmethod
    def parse_backtest_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Replay LSTM OOS predictions via the new model runner.")
        parser.add_argument("--predictions", default=None, type=Path)
        parser.add_argument("--metadata", default=None, type=Path)
        parser.add_argument("--chart", default=None, type=Path)
        parser.add_argument("--start-date", default=None)
        parser.add_argument("--end-date", default=None)
        return parser.parse_args(list(argv or []))

    @staticmethod
    def print_walk_forward_summary(result) -> None:
        if result.predictions.empty:
            print("LSTM walk-forward summary: no predictions.")
            return
        period_start = result.predictions["timestamp"].min()
        period_end = result.predictions["timestamp"].max()
        fold_count = len(result.fold_details)
        avg_predictions = len(result.predictions) / max(fold_count, 1)
        print("=" * 72)
        print("LSTM WALK-FORWARD OOS SUMMARY")
        print(f"Split: {result.request.split_mode} | window={result.request.monthly_window_mode}")
        print(f"Train/test months: {result.request.monthly_train_months}/{result.request.monthly_test_months}")
        print(f"Purge gap: {result.request.purge_gap}")
        print(f"Sequence length: {result.request.sequence_length}")
        print(f"Folds: {fold_count}")
        print(f"Predictions: {len(result.predictions)} | avg/fold={avg_predictions:.1f}")
        print(f"Prediction period: {period_start} -> {period_end}")
        print(f"Features: {len(result.feature_columns)}")
        print("=" * 72)

    @staticmethod
    def print_walk_forward_plan(plan) -> None:
        print("=" * 72)
        print("LSTM WALK-FORWARD PLAN (DRY RUN)")
        print(f"Symbols: {', '.join(plan.symbols)}")
        print(f"Split: {plan.split_mode} | window={plan.monthly_window_mode}")
        print(f"Purge gap: {plan.purge_gap}")
        print(f"Sequence length: {plan.sequence_length}")
        print(f"Sample rows: {plan.sample_rows}")
        print(f"History rows: {plan.history_rows}")
        print(f"Features: {plan.feature_count}")
        print(f"Folds: {len(plan.folds)}")
        print(f"Estimated prediction sequences: {plan.estimated_prediction_rows}")
        print("-" * 72)
        for fold in plan.folds[:5]:
            print(
                "Fold {fold}: train_seq={train_sequences} test_seq={test_sequences} "
                "[{test_start} -> {test_end}]".format(**fold)
            )
        if len(plan.folds) > 5:
            print(f"... {len(plan.folds) - 5} more folds")
        print("=" * 72)
