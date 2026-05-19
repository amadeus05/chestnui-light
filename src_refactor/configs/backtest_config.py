from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src_refactor.configs.loader import load_config_mapping


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    db_path: str = "./data/market_data.db"
    exchange_code: str = "bybit"
    symbols: tuple[str, ...] = ()
    timeframe: str = "1h"
    htf_timeframe: str = "4h"
    start: str | None = None
    end: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MarketDataConfig":
        payload = payload or {}
        defaults = cls()
        return cls(
            db_path=str(payload.get("db_path", defaults.db_path)),
            exchange_code=str(payload.get("exchange_code", defaults.exchange_code)),
            symbols=tuple(str(symbol) for symbol in payload.get("symbols", ())),
            timeframe=str(payload.get("timeframe", defaults.timeframe)),
            htf_timeframe=str(payload.get("htf_timeframe", defaults.htf_timeframe)),
            start=_optional_str(payload.get("start")),
            end=_optional_str(payload.get("end")),
        )


@dataclass(frozen=True, slots=True)
class ModelRunConfig:
    model_type: str = "lightgbm"
    profile: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelRunConfig":
        payload = payload or {}
        defaults = cls()
        return cls(
            model_type=str(payload.get("model_type", defaults.model_type)),
            profile=str(payload.get("profile", defaults.profile)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RuntimeRunConfig:
    initial_balance: float = 100.0
    taker_fee: float = 0.0004
    slippage: float = 0.0003
    risk_per_trade: float = 0.01
    leverage: float = 1.0
    min_position_notional: float = 10.0
    max_open_positions: int = 1
    sl_cooldown_bars: int = 0
    max_sl_per_day: int = 0
    reduce_risk_after_consecutive_losses: int = 0
    reduced_risk_per_trade: float | None = None
    directional_proba_threshold: float = 0.5
    min_signal_gap: float = 0.0
    allow_longs: bool = True
    allow_shorts: bool = True
    max_new_positions_per_bar: int = 1

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "RuntimeRunConfig":
        payload = payload or {}
        defaults = cls()
        pricing = dict(payload.get("pricing") or {})
        risk = dict(payload.get("risk") or {})
        signals = dict(payload.get("signals") or {})
        trading = dict(payload.get("trading") or {})
        return cls(
            initial_balance=float(payload.get("initial_balance", defaults.initial_balance)),
            taker_fee=float(pricing.get("taker_fee", defaults.taker_fee)),
            slippage=float(pricing.get("slippage", defaults.slippage)),
            risk_per_trade=float(risk.get("risk_per_trade", defaults.risk_per_trade)),
            leverage=float(risk.get("leverage", defaults.leverage)),
            min_position_notional=float(risk.get("min_position_notional", defaults.min_position_notional)),
            max_open_positions=int(risk.get("max_open_positions", defaults.max_open_positions)),
            sl_cooldown_bars=int(risk.get("sl_cooldown_bars", defaults.sl_cooldown_bars)),
            max_sl_per_day=int(risk.get("max_sl_per_day", defaults.max_sl_per_day)),
            reduce_risk_after_consecutive_losses=int(
                risk.get(
                    "reduce_risk_after_consecutive_losses",
                    defaults.reduce_risk_after_consecutive_losses,
                )
            ),
            reduced_risk_per_trade=_optional_float(risk.get("reduced_risk_per_trade")),
            directional_proba_threshold=float(
                signals.get("directional_proba_threshold", defaults.directional_proba_threshold)
            ),
            min_signal_gap=float(signals.get("min_signal_gap", defaults.min_signal_gap)),
            allow_longs=bool(signals.get("allow_longs", defaults.allow_longs)),
            allow_shorts=bool(signals.get("allow_shorts", defaults.allow_shorts)),
            max_new_positions_per_bar=int(
                trading.get("max_new_positions_per_bar", defaults.max_new_positions_per_bar)
            ),
        )


@dataclass(frozen=True, slots=True)
class StoredBacktestConfig:
    predictions_path: str | None = None
    model_id: str | None = None
    start: str | None = None
    end: str | None = None
    keep_open_positions: bool = False
    equity_curve_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "StoredBacktestConfig":
        payload = payload or {}
        return cls(
            predictions_path=_optional_str(payload.get("predictions_path")),
            model_id=_optional_str(payload.get("model_id")),
            start=_optional_str(payload.get("start")),
            end=_optional_str(payload.get("end")),
            keep_open_positions=bool(payload.get("keep_open_positions", False)),
            equity_curve_path=_optional_str(payload.get("equity_curve_path")),
        )


@dataclass(frozen=True, slots=True)
class WalkForwardOosConfig:
    predictions_path: str | None = None
    split_mode: str = "monthly_expanding"
    n_splits: int = 5
    train_months: int = 6
    test_months: int = 1
    purge_gap: int = 0
    feature_request: dict[str, Any] | None = None
    feature_profiles: dict[str, Any] | None = None
    labeling: dict[str, Any] | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    keep_open_positions: bool = False
    equity_curve_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "WalkForwardOosConfig":
        payload = payload or {}
        defaults = cls()
        return cls(
            predictions_path=_optional_str(payload.get("predictions_path")),
            split_mode=str(payload.get("split_mode", defaults.split_mode)),
            n_splits=int(payload.get("n_splits", defaults.n_splits)),
            train_months=int(payload.get("train_months", defaults.train_months)),
            test_months=int(payload.get("test_months", defaults.test_months)),
            purge_gap=int(payload.get("purge_gap", defaults.purge_gap)),
            feature_request=_optional_dict(payload.get("feature_request")),
            feature_profiles=_optional_dict(payload.get("feature_profiles")),
            labeling=_optional_dict(payload.get("labeling")),
            model_metadata=dict(payload.get("model_metadata") or {}),
            keep_open_positions=bool(payload.get("keep_open_positions", False)),
            equity_curve_path=_optional_str(payload.get("equity_curve_path")),
        )


@dataclass(frozen=True, slots=True)
class BacktestCliConfig:
    market: MarketDataConfig = field(default_factory=MarketDataConfig)
    model: ModelRunConfig = field(default_factory=ModelRunConfig)
    runtime: RuntimeRunConfig = field(default_factory=RuntimeRunConfig)
    stored: StoredBacktestConfig = field(default_factory=StoredBacktestConfig)
    walk_forward: WalkForwardOosConfig = field(default_factory=WalkForwardOosConfig)

    @classmethod
    def from_path(cls, path: str | Path) -> "BacktestCliConfig":
        return cls.from_mapping(load_config_mapping(path))

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "BacktestCliConfig":
        payload = payload or {}
        return cls(
            market=MarketDataConfig.from_mapping(payload.get("market")),
            model=ModelRunConfig.from_mapping(payload.get("model")),
            runtime=RuntimeRunConfig.from_mapping(payload.get("runtime")),
            stored=StoredBacktestConfig.from_mapping(payload.get("stored_backtest")),
            walk_forward=WalkForwardOosConfig.from_mapping(payload.get("walk_forward")),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected config value to be a mapping/object.")
    return dict(value)
