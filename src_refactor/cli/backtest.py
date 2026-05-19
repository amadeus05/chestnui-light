from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src_refactor.application.backtest import (
    BacktestEquityCurveRenderer,
    BacktestTextReportRenderer,
    StoredPredictionBacktestFlow,
    StoredPredictionBacktestRequest,
    WalkForwardOosBacktestFlow,
)
from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig
from src_refactor.cli.common import (
    arg,
    json_payload,
    loaded_config,
    model_metadata,
    model_spec,
    optional_timestamp,
    repository,
    required,
    runtime_config,
    symbols,
)
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader, HistoricalMarketMapLoader
from src_refactor.infrastructure.predictions import ParquetPredictionStore


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored backtest runtime CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stored = subparsers.add_parser(
        "stored",
        parents=[_common_parser()],
        help="Run a backtest from stored parquet predictions.",
    )
    stored.add_argument("--predictions-path", default=None)
    stored.add_argument("--model-id", default=None)
    stored.add_argument("--start", default=None)
    stored.add_argument("--end", default=None)
    stored.add_argument("--keep-open-positions", action="store_true", default=None)
    stored.add_argument("--equity-curve-path", default=None)
    stored.set_defaults(handler=run_stored_backtest)

    wvf = subparsers.add_parser(
        "wvf-oos",
        parents=[_common_parser()],
        help="Train walk-forward OOS predictions and backtest them through the same runtime.",
    )
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
    wvf.add_argument("--keep-open-positions", action="store_true", default=None)
    wvf.add_argument("--equity-curve-path", default=None)
    wvf.set_defaults(handler=run_walk_forward_oos_backtest)
    return parser


def run_stored_backtest(args: argparse.Namespace) -> None:
    config = loaded_config(args)
    resolved_symbols = symbols(args, config)
    model = model_spec(args, config)
    predictions_path = Path(required(arg(args, "predictions_path", config.stored.predictions_path), "predictions_path"))
    start = optional_timestamp(arg(args, "start", config.stored.start or config.market.start))
    end = optional_timestamp(arg(args, "end", config.stored.end or config.market.end))
    model_id = arg(args, "model_id", config.stored.model_id) or model.model_id
    if start is None:
        start, inferred_end = _prediction_window(predictions_path, model_id=model_id, symbols=resolved_symbols)
        end = end or inferred_end

    market_repository = repository(args, config)
    result = StoredPredictionBacktestFlow(
        config=runtime_config(args, config, model),
        candle_loader=HistoricalCandleFrameLoader(market_repository),
        prediction_store=ParquetPredictionStore(predictions_path),
    ).run(
        StoredPredictionBacktestRequest(
            symbols=resolved_symbols,
            timeframe=arg(args, "timeframe", config.market.timeframe),
            model_id=model_id,
            start=start,
            end=end,
            close_open_positions=not bool(arg(args, "keep_open_positions", config.stored.keep_open_positions)),
        )
    )
    _print_backtest_result(
        result.metrics,
        equity_curve_path=arg(args, "equity_curve_path", config.stored.equity_curve_path),
    )


def run_walk_forward_oos_backtest(args: argparse.Namespace) -> None:
    config = loaded_config(args)
    resolved_symbols = symbols(args, config)
    resolved_model_metadata = model_metadata(args, config)
    model = model_spec(args, config, metadata=resolved_model_metadata)
    market_repository = repository(args, config)
    market_loader = HistoricalMarketMapLoader.from_repository(
        market_repository,
        start=optional_timestamp(arg(args, "start", config.market.start)),
        end=optional_timestamp(arg(args, "end", config.market.end)),
    )
    predictions_path = required(arg(args, "predictions_path", config.walk_forward.predictions_path), "predictions_path")
    result = WalkForwardOosBacktestFlow(
        market_loader=market_loader,
        htf_timeframe=arg(args, "htf_timeframe", config.market.htf_timeframe),
    ).run(
        training_config=WvfOosRunConfig(
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
            model_metadata=model.metadata,
        ),
        runtime_config=runtime_config(args, config, model),
        close_open_positions=not bool(arg(args, "keep_open_positions", config.walk_forward.keep_open_positions)),
    )
    print(f"Saved OOS predictions: {predictions_path}")
    _print_backtest_result(
        result.backtest.metrics,
        equity_curve_path=arg(args, "equity_curve_path", config.walk_forward.equity_curve_path),
    )


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--exchange-code", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--model-type", choices=["lightgbm", "lstm_features", "lstm_candles"], default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--initial-balance", type=float, default=None)
    parser.add_argument("--taker-fee", type=float, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--risk-per-trade", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--min-position-notional", type=float, default=None)
    parser.add_argument("--max-open-positions", type=int, default=None)
    parser.add_argument("--sl-cooldown-bars", type=int, default=None)
    parser.add_argument("--max-sl-per-day", type=int, default=None)
    parser.add_argument("--reduce-risk-after-consecutive-losses", type=int, default=None)
    parser.add_argument("--reduced-risk-per-trade", type=float, default=None)
    parser.add_argument("--directional-proba-threshold", type=float, default=None)
    parser.add_argument("--min-signal-gap", type=float, default=None)
    parser.add_argument("--no-longs", dest="allow_longs", action="store_false", default=None)
    parser.add_argument("--no-shorts", dest="allow_shorts", action="store_false", default=None)
    parser.add_argument("--max-new-positions-per-bar", type=int, default=None)
    return parser


def _prediction_window(
    path: Path,
    *,
    model_id: str,
    symbols: tuple[str, ...],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists():
        return None, None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None, None

    mask = frame["model_id"] == model_id
    if symbols:
        mask &= frame["symbol"].isin(symbols)
    timestamps = pd.to_datetime(frame.loc[mask, "timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None, None
    return timestamps.min(), timestamps.max()


def _print_backtest_result(metrics: Any, *, equity_curve_path: str | None) -> None:
    if metrics is None:
        print("Backtest finished without metrics.")
        return
    print(BacktestTextReportRenderer().render(metrics))
    if equity_curve_path:
        saved = BacktestEquityCurveRenderer().save(metrics, Path(equity_curve_path))
        if saved is not None:
            print(f"Saved equity curve: {saved}")


if __name__ == "__main__":
    main()
