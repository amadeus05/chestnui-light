import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bt
import config as cfg
import train


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest from saved LSTM walk-forward OOS predictions.")
    parser.add_argument(
        "--predictions",
        default=str(cfg.MODELS_DIR / "lstm_walk_forward_oos_predictions.csv"),
        help="CSV with timestamp,symbol,p_short,p_long,fold.",
    )
    parser.add_argument(
        "--features",
        default=str(cfg.MODELS_DIR / "lstm_target_features.json"),
        help="LSTM feature metadata JSON.",
    )
    parser.add_argument(
        "--chart",
        default=str(cfg.BACKTEST_CHARTS_DIR / "equity_curve_lstm_walk_forward.png"),
        help="Output equity curve path.",
    )
    parser.add_argument("--start-date", default=None, help="Optional backtest start timestamp.")
    parser.add_argument("--end-date", default=None, help="Optional backtest end timestamp.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database for raw/features/barriers.")
    return parser.parse_args()


def load_predictions(path: str) -> pd.DataFrame:
    predictions = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "p_short", "p_long"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    return predictions


def filter_predictions_by_period(
    predictions: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = predictions.copy()
    available_start = pd.to_datetime(filtered["timestamp"]).min()
    available_end = pd.to_datetime(filtered["timestamp"]).max()

    if start_date:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(start_ts):
            raise ValueError(f"Invalid --start-date: {start_date}")
        filtered = filtered.loc[filtered["timestamp"] >= start_ts].copy()
    if end_date:
        end_ts = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(end_ts):
            raise ValueError(f"Invalid --end-date: {end_date}")
        filtered = filtered.loc[filtered["timestamp"] <= end_ts].copy()

    if filtered.empty:
        raise ValueError(
            "No predictions in requested period. "
            f"Available prediction range: {available_start} -> {available_end}"
        )

    actual_start = pd.to_datetime(filtered["timestamp"]).min()
    actual_end = pd.to_datetime(filtered["timestamp"]).max()
    print(
        "LSTM prediction replay window: "
        f"{actual_start} -> {actual_end} "
        f"(available: {available_start} -> {available_end})"
    )
    return filtered


def load_features_meta(path: str, predictions: pd.DataFrame) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_columns = payload.get("feature_columns", [])
    symbols = payload.get("symbols") or sorted(predictions["symbol"].dropna().astype(str).unique().tolist())
    return {
        "feature_columns": feature_columns,
        "label_mapping": {"short": 0, "long": 1},
        "inverse_label_mapping": {str(key): value for key, value in train.CLASS_TO_LABEL.items()},
        "symbols": list(symbols),
        "rows": int(len(predictions)),
        "task_type": "lstm_binary_directional_walk_forward_oos_replay",
        "train_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "feature_clip": {
            "enabled": False,
            "bounds": {},
            "note": "Replay of saved LSTM OOS predictions.",
        },
    }


def main():
    args = parse_args()
    predictions = load_predictions(args.predictions)
    predictions = filter_predictions_by_period(predictions, args.start_date, args.end_date)
    features_meta = load_features_meta(args.features, predictions)
    bt.backtest(
        features_meta=features_meta,
        predictions=predictions,
        equity_curve_path=Path(args.chart),
        result_title="LSTM WALK-FORWARD OOS BACKTEST REPLAY",
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
