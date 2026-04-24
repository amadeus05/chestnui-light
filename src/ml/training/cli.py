import argparse

import config as cfg

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
        "--split-mode",
        choices=["tscv", "monthly"],
        default="tscv",
        help="Walk-forward split mode: tscv uses TimeSeriesSplit, monthly uses rolling calendar OOS tests.",
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
        help="Purge gap in timestamps between train and test folds to avoid target leakage.",
    )
    return parser.parse_args()
