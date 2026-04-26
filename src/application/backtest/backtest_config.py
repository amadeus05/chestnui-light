"""Типизированный срез настроек бэктеста (без import *)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestRunConfig:
    """Параметры прогона; по умолчанию читаются из config."""

    symbols: list[str]
    timeframe: str
    htf_timeframe: str
    initial_balance: float
    slippage: float
    taker_com: float
    leverage: float
    risk_per_trade: float
    reduced_risk_per_trade: float
    reduce_risk_after_consecutive_losses: int
    directional_proba_threshold: float
    min_signal_gap: float
    allow_longs: bool
    allow_shorts: bool
    max_new_positions_per_bar: int
    max_open_positions: int
    sl_cooldown_bars: int
    max_sl_per_day: int
    min_position_notional: float
    realtime_features: bool
    charts_dir: Path
    default_equity_chart_path: Path

    @staticmethod
    def from_config_module() -> BacktestRunConfig:
        import config as cfg

        dpt = getattr(
            cfg,
            "DIRECTIONAL_PROBA_THRESHOLD",
            getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5),
        )
        charts = Path(getattr(cfg, "BACKTEST_CHARTS_DIR", "backtest_charts"))
        return BacktestRunConfig(
            symbols=list(cfg.SYMBOLS),
            timeframe=str(cfg.TIMEFRAME),
            htf_timeframe=str(cfg.HTF_TIMEFRAME),
            initial_balance=float(getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)),
            slippage=float(getattr(cfg, "SLIPPAGE", 0.0003)),
            taker_com=float(getattr(cfg, "TAKER_COM", 0.0004)),
            leverage=float(getattr(cfg, "LEVERAGE", 1)),
            risk_per_trade=float(getattr(cfg, "RISK_PER_TRADE", 0.01)),
            reduced_risk_per_trade=float(
                getattr(
                    cfg,
                    "BACKTEST_REDUCED_RISK_PER_TRADE",
                    getattr(cfg, "RISK_PER_TRADE", 0.01),
                )
            ),
            reduce_risk_after_consecutive_losses=int(
                getattr(cfg, "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0)
            ),
            directional_proba_threshold=float(dpt),
            min_signal_gap=float(getattr(cfg, "MIN_SIGNAL_GAP", 0.0)),
            allow_longs=bool(getattr(cfg, "ALLOW_LONGS", True)),
            allow_shorts=bool(getattr(cfg, "ALLOW_SHORTS", True)),
            max_new_positions_per_bar=int(getattr(cfg, "BACKTEST_MAX_NEW_POSITIONS_PER_BAR", 1)),
            max_open_positions=int(getattr(cfg, "BACKTEST_MAX_OPEN_POSITIONS", 5)),
            sl_cooldown_bars=int(getattr(cfg, "BACKTEST_SL_COOLDOWN_BARS", 0)),
            max_sl_per_day=int(getattr(cfg, "BACKTEST_MAX_SL_PER_DAY", 0)),
            min_position_notional=float(getattr(cfg, "BACKTEST_MIN_POSITION_NOTIONAL", 10.0)),
            realtime_features=bool(getattr(cfg, "BACKTEST_REALTIME_FEATURES", False)),
            charts_dir=charts,
            default_equity_chart_path=charts / "equity_curve.png",
        )
