"""CLI: walk-forward OOS-предикты и портфельный бэктест."""
import argparse

import config as cfg
from src.application.walk_forward_oos import run_walk_forward_oos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portfolio backtest using only walk-forward out-of-sample predictions."
    )
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to load, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument(
        "--split-mode",
        choices=["tscv", "monthly"],
        default="tscv",
        help="Walk-forward split mode: tscv uses TimeSeriesSplit, monthly uses 1-month rolling OOS tests.",
    )
    parser.add_argument(
        "--monthly-train-months",
        type=int,
        default=6,
        help="Training window in months for --split-mode monthly.",
    )
    parser.add_argument(
        "--monthly-window-mode",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Monthly WFV train mode: expanding uses all history, rolling uses only the latest train window.",
    )
    parser.add_argument(
        "--monthly-test-months",
        type=int,
        default=1,
        help="Test window size in months for --split-mode monthly.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=12,
        help="Purge gap in timestamps between train and test folds.",
    )
    parser.add_argument(
        "--predictions-name",
        default="walk_forward_oos_predictions",
        help="Base filename for saved OOS predictions.",
    )
    return parser.parse_args()


def main() -> None:
    run_walk_forward_oos(parse_args())


if __name__ == "__main__":
    main()
