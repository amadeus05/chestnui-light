from __future__ import annotations

import argparse
from typing import Sequence

import config as cfg
from src.models.base import BaseModelRunner
from src.models.lightgbm.artifacts import LightGbmArtifactWriter
from src.models.lightgbm.trainer import LightGbmTrainer, LightGbmTrainRequest


class LightGbmRunner(BaseModelRunner):
    """Runner for the new LightGBM production path plus legacy WFV/replay."""

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_new_production_train(argv)

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
