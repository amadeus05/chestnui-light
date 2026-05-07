from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import config as cfg
from src.models.base import BaseModelRunner
from src.models.lightgbm.artifacts import LightGbmArtifactWriter
from src.models.lightgbm.backtest import LightGbmBacktestRequest, LightGbmBacktestRunner
from src.models.lightgbm.trainer import LightGbmTrainer, LightGbmTrainRequest
from src.models.lightgbm.walk_forward import LightGbmWalkForwardRequest, LightGbmWalkForwardRunner


class LightGbmRunner(BaseModelRunner):
    """Runner for the new LightGBM production path plus legacy WFV/replay."""

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_walk_forward(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_walk_forward_args(argv)
        result = LightGbmWalkForwardRunner().run(
            LightGbmWalkForwardRequest(
                db_path=args.db_path,
                symbols=args.symbols,
                seed=args.seed,
                n_splits=args.n_splits,
                split_mode=args.split_mode,
                monthly_train_months=args.monthly_train_months,
                monthly_test_months=args.monthly_test_months,
                monthly_window_mode=args.monthly_window_mode,
                purge_gap=args.purge_gap,
            )
        )
        paths = LightGbmArtifactWriter(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).save_walk_forward_result(result)
        print(f"Saved '{self.spec.key}' walk-forward artifacts to {paths.root}")

    def run_backtest(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_backtest_args(argv)
        LightGbmBacktestRunner(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).run(
            LightGbmBacktestRequest(
                predictions_path=args.predictions,
                metadata_path=args.metadata,
                chart_path=args.chart,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        )

    def run_new_production_train(self, argv: Sequence[str] | None = None) -> None:
        args = self.parse_train_args(argv)
        trained = LightGbmTrainer().train_production(
            LightGbmTrainRequest(
                db_path=args.db_path,
                symbols=args.symbols,
                seed=args.seed,
                n_estimators=args.n_estimators,
            )
        )
        paths = LightGbmArtifactWriter(
            spec=self.spec,
            artifact_store=self.artifact_store,
        ).save_production_model(trained)
        print(f"Saved '{self.spec.key}' production artifacts to {paths.root}")

    @staticmethod
    def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train production LightGBM model via the new model runner.")
        parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
        parser.add_argument("--symbols", nargs="+", default=list(cfg.SYMBOLS), help="Symbols to load.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--n-estimators", type=int, default=800, help="Number of production estimators.")
        return parser.parse_args(list(argv or []))

    @staticmethod
    def parse_walk_forward_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Build LightGBM walk-forward OOS predictions via the new model runner.")
        parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
        parser.add_argument("--symbols", nargs="+", default=list(cfg.SYMBOLS), help="Symbols to load.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward folds.")
        parser.add_argument("--split-mode", choices=["tscv", "monthly"], default="tscv")
        parser.add_argument("--monthly-train-months", type=int, default=6)
        parser.add_argument("--monthly-test-months", type=int, default=1)
        parser.add_argument("--monthly-window-mode", choices=["expanding", "rolling"], default="expanding")
        parser.add_argument("--purge-gap", type=int, default=cfg.effective_max_label_horizon())
        return parser.parse_args(list(argv or []))

    @staticmethod
    def parse_backtest_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Replay LightGBM OOS predictions via the new model runner.")
        parser.add_argument("--predictions", default=None, type=Path)
        parser.add_argument("--metadata", default=None, type=Path)
        parser.add_argument("--chart", default=None, type=Path)
        parser.add_argument("--start-date", default=None)
        parser.add_argument("--end-date", default=None)
        return parser.parse_args(list(argv or []))
