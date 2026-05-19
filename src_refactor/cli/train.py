from __future__ import annotations

import argparse
from pathlib import Path

from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig, run_wvf_oos
from src_refactor.cli.common import (
    add_config_args,
    add_market_args,
    add_model_args,
    add_walk_forward_args,
    arg,
    json_payload,
    loaded_config,
    model_metadata,
    optional_timestamp,
    repository,
    required,
    symbols,
)
from src_refactor.infrastructure.feeds import HistoricalMarketMapLoader


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
    add_config_args(wvf)
    add_market_args(wvf)
    add_model_args(wvf)
    add_walk_forward_args(wvf)
    wvf.set_defaults(handler=run_walk_forward_oos_training)
    return parser


def run_walk_forward_oos_training(args: argparse.Namespace) -> None:
    config = loaded_config(args)
    resolved_symbols = symbols(args, config)
    resolved_model_metadata = model_metadata(args, config)
    predictions_path = required(arg(args, "predictions_path", config.walk_forward.predictions_path), "predictions_path")
    market_repository = repository(args, config)
    market_loader = HistoricalMarketMapLoader.from_repository(
        market_repository,
        start=optional_timestamp(arg(args, "start", config.market.start)),
        end=optional_timestamp(arg(args, "end", config.market.end)),
    )
    result = run_wvf_oos(
        base_candle_map=market_loader.load_base_map(
            resolved_symbols,
            arg(args, "timeframe", config.market.timeframe),
        ),
        htf_candle_map=market_loader.load_htf_map(
            resolved_symbols,
            arg(args, "htf_timeframe", config.market.htf_timeframe),
        ),
        config=WvfOosRunConfig(
            model_type=arg(args, "model_type", config.model.model_type),
            timeframe=arg(args, "timeframe", config.market.timeframe),
            profile=arg(args, "profile", config.model.profile),
            symbols=resolved_symbols,
            split_mode=arg(args, "split_mode", config.walk_forward.split_mode),
            n_splits=arg(args, "n_splits", config.walk_forward.n_splits),
            train_months=arg(args, "train_months", config.walk_forward.train_months),
            test_months=arg(args, "test_months", config.walk_forward.test_months),
            purge_gap=arg(args, "purge_gap", config.walk_forward.purge_gap),
            predictions_path=Path(predictions_path),
            feature_request=json_payload(args.feature_request_json) or config.walk_forward.feature_request,
            feature_profiles=json_payload(args.feature_profiles_json) or config.walk_forward.feature_profiles,
            labeling_config=json_payload(args.labeling_json) or config.walk_forward.labeling,
            model_metadata=resolved_model_metadata,
        ),
    )
    print(f"Model: {result.config.model.model_id}")
    print(f"Folds: {len(result.folds)}")
    print(f"Predictions: {len(result.predictions)}")
    print(f"Saved OOS predictions: {result.prediction_store_path}")


if __name__ == "__main__":
    main()
