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
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest from saved candle-LSTM walk-forward OOS predictions.")
    parser.add_argument(
        "--predictions",
        default=str(cfg.MODELS_DIR / "lstm_candles_walk_forward_oos_predictions.csv"),
        help="CSV with timestamp,symbol,p_short,p_long,fold.",
    )
    parser.add_argument(
        "--features",
        default=str(cfg.MODELS_DIR / "lstm_candles_target_features.json"),
        help="Feature metadata JSON.",
    )
    parser.add_argument(
        "--chart",
        default=str(cfg.BACKTEST_CHARTS_DIR / "equity_curve_lstm_candles_walk_forward.png"),
        help="Output equity curve path.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database for raw/features/barriers.")
    return parser.parse_args()


def load_predictions(path: str) -> pd.DataFrame:
    predictions = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "p_short", "p_long"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    return predictions


def attach_barrier_columns_from_feature_tables(predictions: pd.DataFrame, db_path: str | None = None) -> pd.DataFrame:
    if {"barrier_stop_pct", "barrier_take_pct"}.issubset(predictions.columns):
        return predictions

    symbols = sorted(predictions["symbol"].dropna().astype(str).unique().tolist())
    repository = HistoricalKlineRepository(db_path=db_path or cfg.DB_PATH)
    barrier_frames = []
    for symbol in symbols:
        feature_frame = repository.load_features(symbol)
        if feature_frame.empty:
            continue
        required_columns = {"timestamp", "symbol", "barrier_stop_pct", "barrier_take_pct"}
        missing = required_columns.difference(feature_frame.columns)
        if missing:
            continue
        barrier_frames.append(
            feature_frame[["timestamp", "symbol", "barrier_stop_pct", "barrier_take_pct"]].copy()
        )
    if not barrier_frames:
        raise ValueError("No feature-table barrier rows found for candle-LSTM predictions.")

    barrier_frame = pd.concat(barrier_frames, ignore_index=True)
    barrier_frame["timestamp"] = pd.to_datetime(barrier_frame["timestamp"])
    barrier_frame = barrier_frame.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    merged = predictions.merge(
        barrier_frame,
        on=["timestamp", "symbol"],
        how="left",
    )
    missing_barriers = merged["barrier_stop_pct"].isna().sum() + merged["barrier_take_pct"].isna().sum()
    if missing_barriers:
        raise ValueError(
            "Failed to attach barrier columns to candle-LSTM predictions. "
            f"Missing merged barrier values: {int(missing_barriers)}"
        )
    return merged


def filter_predictions_by_period(predictions: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    filtered = predictions.copy()
    if start_date:
        filtered = filtered.loc[filtered["timestamp"] >= pd.to_datetime(start_date, errors="raise")].copy()
    if end_date:
        filtered = filtered.loc[filtered["timestamp"] <= pd.to_datetime(end_date, errors="raise")].copy()
    if filtered.empty:
        raise ValueError("No predictions remain after date filters.")
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
        "task_type": "lstm_candles_binary_directional_walk_forward_oos_replay",
        "train_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "feature_clip": {
            "enabled": False,
            "bounds": {},
            "note": "Replay of saved candle-LSTM OOS predictions.",
        },
    }


def main():
    args = parse_args()
    predictions = load_predictions(args.predictions)
    predictions = attach_barrier_columns_from_feature_tables(predictions, db_path=args.db_path)
    predictions = filter_predictions_by_period(predictions, args.start_date, args.end_date)
    features_meta = load_features_meta(args.features, predictions)
    bt.backtest(
        features_meta=features_meta,
        predictions=predictions,
        equity_curve_path=Path(args.chart),
        result_title="LSTM CANDLES WALK-FORWARD OOS BACKTEST REPLAY",
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
