from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.configs import BacktestCliConfig
from src_refactor.core.types import ModelSpec
from src_refactor.domain.execution import ExecutionPricingConfig
from src_refactor.domain.risk.risk_manager import RiskConfig
from src_refactor.domain.signals import SignalProcessingConfig
from src_refactor.domain.trading import TradingEngineConfig
from src_refactor.infrastructure.market_data import ParquetMarketDataStore


MODEL_TYPE_CHOICES = ("lightgbm", "lstm_features", "lstm_candles")
SPLIT_MODE_CHOICES = ("tscv", "monthly_expanding", "monthly_rolling")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)


def add_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--exchange-code", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframe", default=None)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-type", choices=MODEL_TYPE_CHOICES, default=None)
    parser.add_argument("--profile", default=None)


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
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


def add_walk_forward_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--htf-timeframe", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--split-mode", choices=SPLIT_MODE_CHOICES, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--purge-gap", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--feature-request-json", default=None)
    parser.add_argument("--feature-profiles-json", default=None)
    parser.add_argument("--labeling-json", default=None)
    parser.add_argument("--model-metadata-json", default=None)


def add_stored_prediction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)


def add_backtest_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keep-open-positions", action="store_true", default=None)
    parser.add_argument("--equity-curve-path", default=None)


def loaded_config(args: argparse.Namespace) -> BacktestCliConfig:
    return BacktestCliConfig.from_path(args.config) if args.config else BacktestCliConfig()


def symbols(args: argparse.Namespace, config: BacktestCliConfig) -> tuple[str, ...]:
    resolved = tuple(arg(args, "symbols", config.market.symbols) or ())
    if not resolved:
        raise ValueError("symbols must be provided via --symbols or market.symbols in config.")
    return resolved


def arg(args: argparse.Namespace, name: str, fallback: Any) -> Any:
    value = getattr(args, name, None)
    return fallback if value is None else value


def required(value: Any, label: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"{label} must be provided via CLI or config.")
    return value


def json_payload(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


def repository(args: argparse.Namespace, config: BacktestCliConfig) -> ParquetMarketDataStore:
    return ParquetMarketDataStore(
        root=arg(args, "data_root", config.market.data_root),
        exchange_code=arg(args, "exchange_code", config.market.exchange_code),
    )


def model_metadata(args: argparse.Namespace, config: BacktestCliConfig) -> dict[str, Any]:
    return {
        **config.model.metadata,
        **config.walk_forward.model_metadata,
        **(json_payload(getattr(args, "model_metadata_json", None)) or {}),
    }


def model_spec(
    args: argparse.Namespace,
    config: BacktestCliConfig,
    metadata: dict[str, Any] | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_type=arg(args, "model_type", config.model.model_type),
        timeframe=arg(args, "timeframe", config.market.timeframe),
        profile=arg(args, "profile", config.model.profile),
        symbols=symbols(args, config),
        metadata=metadata if metadata is not None else config.model.metadata,
    )


def runtime_config(args: argparse.Namespace, config: BacktestCliConfig, model: ModelSpec) -> RuntimeConfig:
    runtime = config.runtime
    return RuntimeConfig(
        mode=TradingMode.BACKTEST,
        model=model,
        initial_balance=arg(args, "initial_balance", runtime.initial_balance),
        symbols=symbols(args, config),
        pricing=ExecutionPricingConfig(
            taker_fee=arg(args, "taker_fee", runtime.taker_fee),
            slippage=arg(args, "slippage", runtime.slippage),
        ),
        risk=RiskConfig(
            risk_per_trade=arg(args, "risk_per_trade", runtime.risk_per_trade),
            leverage=arg(args, "leverage", runtime.leverage),
            min_position_notional=arg(args, "min_position_notional", runtime.min_position_notional),
            max_open_positions=arg(args, "max_open_positions", runtime.max_open_positions),
            sl_cooldown_bars=arg(args, "sl_cooldown_bars", runtime.sl_cooldown_bars),
            max_sl_per_day=arg(args, "max_sl_per_day", runtime.max_sl_per_day),
            reduce_risk_after_consecutive_losses=arg(
                args,
                "reduce_risk_after_consecutive_losses",
                runtime.reduce_risk_after_consecutive_losses,
            ),
            reduced_risk_per_trade=arg(args, "reduced_risk_per_trade", runtime.reduced_risk_per_trade),
        ),
        signals=SignalProcessingConfig(
            directional_proba_threshold=arg(
                args,
                "directional_proba_threshold",
                runtime.directional_proba_threshold,
            ),
            min_signal_gap=arg(args, "min_signal_gap", runtime.min_signal_gap),
            allow_longs=arg(args, "allow_longs", runtime.allow_longs),
            allow_shorts=arg(args, "allow_shorts", runtime.allow_shorts),
        ),
        trading=TradingEngineConfig(
            max_new_positions_per_bar=arg(
                args,
                "max_new_positions_per_bar",
                runtime.max_new_positions_per_bar,
            )
        ),
    )
