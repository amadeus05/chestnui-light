from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig, run_wvf_oos
from src_refactor.configs import BacktestCliConfig
from src_refactor.infrastructure.feeds import HistoricalMarketMapLoader
from src_refactor.infrastructure.persistence import SqliteMarketRepository


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored training CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wvf = subparsers.add_parser(
        "wvf-oos",
        help="Train walk-forward OOS predictions without running a backtest.",
    )
    wvf.add_argument("--config", default=None)
    wvf.add_argument("--db-path", default=None)
    wvf.add_argument("--exchange-code", default=None)
    wvf.add_argument("--symbols", nargs="+", default=None)
    wvf.add_argument("--model-type", choices=["lightgbm", "lstm_features", "lstm_candles"], default=None)
    wvf.add_argument("--profile", default=None)
    wvf.add_argument("--timeframe", default=None)
    wvf.add_argument("--htf-timeframe", default=None)
    wvf.add_argument("--predictions-path", default=None)
    wvf.add_argument("--split-mode", choices=["tscv", "monthly_expanding", "monthly_rolling"], default=None)
    wvf.add_argument("--n-splits", type=int, default=None)
    wvf.add_argument("--train-months", type=int, default=None)
    wvf.add_argument("--test-months", type=int, default=None)
    wvf.add_argument("--purge-gap", type=int, default=None)
    wvf.add_argument("--start", default=None)
    wvf.add_argument("--end", default=None)
    wvf.add_argument("--feature-request-json", default=None)
    wvf.add_argument("--feature-profiles-json", default=None)
    wvf.add_argument("--labeling-json", default=None)
    wvf.add_argument("--model-metadata-json", default=None)
    wvf.set_defaults(handler=run_walk_forward_oos_training)
    return parser


def run_walk_forward_oos_training(args: argparse.Namespace) -> None:
    config = _loaded_config(args)
    symbols = _symbols(args, config)
    model_metadata = {
        **config.model.metadata,
        **config.walk_forward.model_metadata,
        **(_json_payload(args.model_metadata_json) or {}),
    }
    predictions_path = _required(_arg(args, "predictions_path", config.walk_forward.predictions_path), "predictions_path")
    repository = SqliteMarketRepository(
        db_path=_arg(args, "db_path", config.market.db_path),
        exchange_code=_arg(args, "exchange_code", config.market.exchange_code),
    )
    market_loader = HistoricalMarketMapLoader.from_repository(
        repository,
        start=_optional_timestamp(_arg(args, "start", config.market.start)),
        end=_optional_timestamp(_arg(args, "end", config.market.end)),
    )
    result = run_wvf_oos(
        base_candle_map=market_loader.load_base_map(symbols, _arg(args, "timeframe", config.market.timeframe)),
        htf_candle_map=market_loader.load_htf_map(symbols, _arg(args, "htf_timeframe", config.market.htf_timeframe)),
        config=WvfOosRunConfig(
            model_type=_arg(args, "model_type", config.model.model_type),
            timeframe=_arg(args, "timeframe", config.market.timeframe),
            profile=_arg(args, "profile", config.model.profile),
            symbols=symbols,
            split_mode=_arg(args, "split_mode", config.walk_forward.split_mode),
            n_splits=_arg(args, "n_splits", config.walk_forward.n_splits),
            train_months=_arg(args, "train_months", config.walk_forward.train_months),
            test_months=_arg(args, "test_months", config.walk_forward.test_months),
            purge_gap=_arg(args, "purge_gap", config.walk_forward.purge_gap),
            predictions_path=Path(predictions_path),
            feature_request=_json_payload(args.feature_request_json) or config.walk_forward.feature_request,
            feature_profiles=_json_payload(args.feature_profiles_json) or config.walk_forward.feature_profiles,
            labeling_config=_json_payload(args.labeling_json) or config.walk_forward.labeling,
            model_metadata=model_metadata,
        ),
    )
    print(f"Model: {result.config.model.model_id}")
    print(f"Folds: {len(result.folds)}")
    print(f"Predictions: {len(result.predictions)}")
    print(f"Saved OOS predictions: {result.prediction_store_path}")


def _loaded_config(args: argparse.Namespace) -> BacktestCliConfig:
    return BacktestCliConfig.from_path(args.config) if args.config else BacktestCliConfig()


def _symbols(args: argparse.Namespace, config: BacktestCliConfig) -> tuple[str, ...]:
    symbols = tuple(_arg(args, "symbols", config.market.symbols) or ())
    if not symbols:
        raise ValueError("symbols must be provided via --symbols or market.symbols in config.")
    return symbols


def _arg(args: argparse.Namespace, name: str, fallback: Any) -> Any:
    value = getattr(args, name, None)
    return fallback if value is None else value


def _required(value: Any, label: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"{label} must be provided via CLI or config.")
    return value


def _json_payload(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


if __name__ == "__main__":
    main()
