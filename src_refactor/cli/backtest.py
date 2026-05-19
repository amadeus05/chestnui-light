from __future__ import annotations

import argparse
import json
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
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig
from src_refactor.core.types import ModelSpec
from src_refactor.domain.execution import ExecutionPricingConfig
from src_refactor.domain.risk.risk_manager import RiskConfig
from src_refactor.domain.signals import SignalProcessingConfig
from src_refactor.domain.trading import TradingEngineConfig
from src_refactor.infrastructure.feeds import HistoricalCandleFrameLoader, HistoricalMarketMapLoader
from src_refactor.infrastructure.persistence import SqliteMarketRepository
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
    stored.add_argument("--predictions-path", required=True)
    stored.add_argument("--model-id", default=None)
    stored.add_argument("--start", default=None)
    stored.add_argument("--end", default=None)
    stored.add_argument("--keep-open-positions", action="store_true")
    stored.add_argument("--equity-curve-path", default=None)
    stored.set_defaults(handler=run_stored_backtest)

    wvf = subparsers.add_parser(
        "wvf-oos",
        parents=[_common_parser()],
        help="Train walk-forward OOS predictions and backtest them through the same runtime.",
    )
    wvf.add_argument("--htf-timeframe", default="4h")
    wvf.add_argument("--predictions-path", required=True)
    wvf.add_argument("--split-mode", choices=["tscv", "monthly_expanding", "monthly_rolling"], default="monthly_expanding")
    wvf.add_argument("--n-splits", type=int, default=5)
    wvf.add_argument("--train-months", type=int, default=6)
    wvf.add_argument("--test-months", type=int, default=1)
    wvf.add_argument("--purge-gap", type=int, default=0)
    wvf.add_argument("--start", default=None)
    wvf.add_argument("--end", default=None)
    wvf.add_argument("--feature-request-json", default=None)
    wvf.add_argument("--feature-profiles-json", default=None)
    wvf.add_argument("--labeling-json", default=None)
    wvf.add_argument("--model-metadata-json", default=None)
    wvf.add_argument("--keep-open-positions", action="store_true")
    wvf.add_argument("--equity-curve-path", default=None)
    wvf.set_defaults(handler=run_walk_forward_oos_backtest)
    return parser


def run_stored_backtest(args: argparse.Namespace) -> None:
    model = _model_spec(args)
    predictions_path = Path(args.predictions_path)
    start = _optional_timestamp(args.start)
    end = _optional_timestamp(args.end)
    model_id = args.model_id or model.model_id
    if start is None:
        start, inferred_end = _prediction_window(predictions_path, model_id=model_id, symbols=tuple(args.symbols))
        end = end or inferred_end

    repository = _repository(args)
    result = StoredPredictionBacktestFlow(
        config=_runtime_config(args, model),
        candle_loader=HistoricalCandleFrameLoader(repository),
        prediction_store=ParquetPredictionStore(predictions_path),
    ).run(
        StoredPredictionBacktestRequest(
            symbols=tuple(args.symbols),
            timeframe=args.timeframe,
            model_id=model_id,
            start=start,
            end=end,
            close_open_positions=not args.keep_open_positions,
        )
    )
    _print_backtest_result(result.metrics, equity_curve_path=args.equity_curve_path)


def run_walk_forward_oos_backtest(args: argparse.Namespace) -> None:
    model = _model_spec(args, metadata=_json_payload(args.model_metadata_json))
    repository = _repository(args)
    market_loader = HistoricalMarketMapLoader.from_repository(
        repository,
        start=_optional_timestamp(args.start),
        end=_optional_timestamp(args.end),
    )
    result = WalkForwardOosBacktestFlow(
        market_loader=market_loader,
        htf_timeframe=args.htf_timeframe,
    ).run(
        training_config=WvfOosRunConfig(
            model_type=args.model_type,
            timeframe=args.timeframe,
            profile=args.profile,
            symbols=tuple(args.symbols),
            split_mode=args.split_mode,
            n_splits=args.n_splits,
            train_months=args.train_months,
            test_months=args.test_months,
            purge_gap=args.purge_gap,
            predictions_path=Path(args.predictions_path),
            feature_request=_json_payload(args.feature_request_json),
            feature_profiles=_json_payload(args.feature_profiles_json),
            labeling_config=_json_payload(args.labeling_json),
            model_metadata=model.metadata,
        ),
        runtime_config=_runtime_config(args, model),
        close_open_positions=not args.keep_open_positions,
    )
    print(f"Saved OOS predictions: {args.predictions_path}")
    _print_backtest_result(result.backtest.metrics, equity_curve_path=args.equity_curve_path)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db-path", default="./data/market_data.db")
    parser.add_argument("--exchange-code", default="bybit")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--model-type", choices=["lightgbm", "lstm_features", "lstm_candles"], default="lightgbm")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--taker-fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--min-position-notional", type=float, default=10.0)
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--sl-cooldown-bars", type=int, default=0)
    parser.add_argument("--max-sl-per-day", type=int, default=0)
    parser.add_argument("--reduce-risk-after-consecutive-losses", type=int, default=0)
    parser.add_argument("--reduced-risk-per-trade", type=float, default=None)
    parser.add_argument("--directional-proba-threshold", type=float, default=0.5)
    parser.add_argument("--min-signal-gap", type=float, default=0.0)
    parser.add_argument("--no-longs", action="store_true")
    parser.add_argument("--no-shorts", action="store_true")
    parser.add_argument("--max-new-positions-per-bar", type=int, default=1)
    return parser


def _runtime_config(args: argparse.Namespace, model: ModelSpec) -> RuntimeConfig:
    return RuntimeConfig(
        mode=TradingMode.BACKTEST,
        model=model,
        initial_balance=args.initial_balance,
        symbols=tuple(args.symbols),
        pricing=ExecutionPricingConfig(taker_fee=args.taker_fee, slippage=args.slippage),
        risk=RiskConfig(
            risk_per_trade=args.risk_per_trade,
            leverage=args.leverage,
            min_position_notional=args.min_position_notional,
            max_open_positions=args.max_open_positions,
            sl_cooldown_bars=args.sl_cooldown_bars,
            max_sl_per_day=args.max_sl_per_day,
            reduce_risk_after_consecutive_losses=args.reduce_risk_after_consecutive_losses,
            reduced_risk_per_trade=args.reduced_risk_per_trade,
        ),
        signals=SignalProcessingConfig(
            directional_proba_threshold=args.directional_proba_threshold,
            min_signal_gap=args.min_signal_gap,
            allow_longs=not args.no_longs,
            allow_shorts=not args.no_shorts,
        ),
        trading=TradingEngineConfig(max_new_positions_per_bar=args.max_new_positions_per_bar),
    )


def _model_spec(args: argparse.Namespace, metadata: dict[str, Any] | None = None) -> ModelSpec:
    return ModelSpec(
        model_type=args.model_type,
        timeframe=args.timeframe,
        profile=args.profile,
        symbols=tuple(args.symbols),
        metadata=metadata or {},
    )


def _repository(args: argparse.Namespace) -> SqliteMarketRepository:
    return SqliteMarketRepository(db_path=args.db_path, exchange_code=args.exchange_code)


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
