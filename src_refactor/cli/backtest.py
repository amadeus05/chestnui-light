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
from src_refactor.configs import BacktestCliConfig
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
    config = _loaded_config(args)
    symbols = _symbols(args, config)
    model = _model_spec(args, config)
    predictions_path = Path(_required(_arg(args, "predictions_path", config.stored.predictions_path), "predictions_path"))
    start = _optional_timestamp(_arg(args, "start", config.stored.start or config.market.start))
    end = _optional_timestamp(_arg(args, "end", config.stored.end or config.market.end))
    model_id = _arg(args, "model_id", config.stored.model_id) or model.model_id
    if start is None:
        start, inferred_end = _prediction_window(predictions_path, model_id=model_id, symbols=symbols)
        end = end or inferred_end

    repository = _repository(args, config)
    result = StoredPredictionBacktestFlow(
        config=_runtime_config(args, config, model),
        candle_loader=HistoricalCandleFrameLoader(repository),
        prediction_store=ParquetPredictionStore(predictions_path),
    ).run(
        StoredPredictionBacktestRequest(
            symbols=symbols,
            timeframe=_arg(args, "timeframe", config.market.timeframe),
            model_id=model_id,
            start=start,
            end=end,
            close_open_positions=not bool(_arg(args, "keep_open_positions", config.stored.keep_open_positions)),
        )
    )
    _print_backtest_result(
        result.metrics,
        equity_curve_path=_arg(args, "equity_curve_path", config.stored.equity_curve_path),
    )


def run_walk_forward_oos_backtest(args: argparse.Namespace) -> None:
    config = _loaded_config(args)
    symbols = _symbols(args, config)
    model_metadata = {
        **config.model.metadata,
        **config.walk_forward.model_metadata,
        **(_json_payload(args.model_metadata_json) or {}),
    }
    model = _model_spec(args, config, metadata=model_metadata)
    repository = _repository(args, config)
    market_loader = HistoricalMarketMapLoader.from_repository(
        repository,
        start=_optional_timestamp(_arg(args, "start", config.market.start)),
        end=_optional_timestamp(_arg(args, "end", config.market.end)),
    )
    predictions_path = _required(_arg(args, "predictions_path", config.walk_forward.predictions_path), "predictions_path")
    result = WalkForwardOosBacktestFlow(
        market_loader=market_loader,
        htf_timeframe=_arg(args, "htf_timeframe", config.market.htf_timeframe),
    ).run(
        training_config=WvfOosRunConfig(
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
            model_metadata=model.metadata,
        ),
        runtime_config=_runtime_config(args, config, model),
        close_open_positions=not bool(_arg(args, "keep_open_positions", config.walk_forward.keep_open_positions)),
    )
    print(f"Saved OOS predictions: {predictions_path}")
    _print_backtest_result(
        result.backtest.metrics,
        equity_curve_path=_arg(args, "equity_curve_path", config.walk_forward.equity_curve_path),
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


def _runtime_config(args: argparse.Namespace, config: BacktestCliConfig, model: ModelSpec) -> RuntimeConfig:
    runtime = config.runtime
    return RuntimeConfig(
        mode=TradingMode.BACKTEST,
        model=model,
        initial_balance=_arg(args, "initial_balance", runtime.initial_balance),
        symbols=_symbols(args, config),
        pricing=ExecutionPricingConfig(
            taker_fee=_arg(args, "taker_fee", runtime.taker_fee),
            slippage=_arg(args, "slippage", runtime.slippage),
        ),
        risk=RiskConfig(
            risk_per_trade=_arg(args, "risk_per_trade", runtime.risk_per_trade),
            leverage=_arg(args, "leverage", runtime.leverage),
            min_position_notional=_arg(args, "min_position_notional", runtime.min_position_notional),
            max_open_positions=_arg(args, "max_open_positions", runtime.max_open_positions),
            sl_cooldown_bars=_arg(args, "sl_cooldown_bars", runtime.sl_cooldown_bars),
            max_sl_per_day=_arg(args, "max_sl_per_day", runtime.max_sl_per_day),
            reduce_risk_after_consecutive_losses=_arg(
                args,
                "reduce_risk_after_consecutive_losses",
                runtime.reduce_risk_after_consecutive_losses,
            ),
            reduced_risk_per_trade=_arg(args, "reduced_risk_per_trade", runtime.reduced_risk_per_trade),
        ),
        signals=SignalProcessingConfig(
            directional_proba_threshold=_arg(
                args,
                "directional_proba_threshold",
                runtime.directional_proba_threshold,
            ),
            min_signal_gap=_arg(args, "min_signal_gap", runtime.min_signal_gap),
            allow_longs=_arg(args, "allow_longs", runtime.allow_longs),
            allow_shorts=_arg(args, "allow_shorts", runtime.allow_shorts),
        ),
        trading=TradingEngineConfig(
            max_new_positions_per_bar=_arg(
                args,
                "max_new_positions_per_bar",
                runtime.max_new_positions_per_bar,
            )
        ),
    )


def _model_spec(
    args: argparse.Namespace,
    config: BacktestCliConfig,
    metadata: dict[str, Any] | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_type=_arg(args, "model_type", config.model.model_type),
        timeframe=_arg(args, "timeframe", config.market.timeframe),
        profile=_arg(args, "profile", config.model.profile),
        symbols=_symbols(args, config),
        metadata=metadata if metadata is not None else config.model.metadata,
    )


def _repository(args: argparse.Namespace, config: BacktestCliConfig) -> SqliteMarketRepository:
    return SqliteMarketRepository(
        db_path=_arg(args, "db_path", config.market.db_path),
        exchange_code=_arg(args, "exchange_code", config.market.exchange_code),
    )


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
