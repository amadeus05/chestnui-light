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
from src_refactor.infrastructure.persistence import SqliteMarketRepository


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


def repository(args: argparse.Namespace, config: BacktestCliConfig) -> SqliteMarketRepository:
    return SqliteMarketRepository(
        db_path=arg(args, "db_path", config.market.db_path),
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
