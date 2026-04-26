"""TradingEngine + PortfolioReplayFeed: дымовой прогон без ML и без входов."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.application.backtest.backtest_config import BacktestRunConfig
from src.application.backtest.portfolio_backtest_context import PortfolioBacktestContext
from src.application.backtest import backtest_data as ld
from src.application.executors.simulation_executor import SimulationExecutor
from src.application.feeds.portfolio_replay_feed import PortfolioReplayFeed
from src.application.trading_engine import TradingEngine
from src.contracts.clock import ReplayClock
from src.domain.execution.execution_service import ExecutionService
from src.domain.execution.models.constraints import BarEntryConstraints
from src.domain.portfolio.portfolio_manager import PortfolioManager
from src.domain.risk.models.risk_limits import RiskLimits
from src.domain.risk.risk_manager import RiskManager
from src.domain.signals.signal_brain import SignalBrain


def test_trading_engine_portfolio_replay_smoke():
    clock = ReplayClock()
    ts0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    ts2 = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")

    main = pd.DataFrame(
        {
            "timestamp": [ts0, ts1, ts2],
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [100.0, 100.0, 100.0],
            "close_time": [ts0 + pd.Timedelta(hours=1), ts1 + pd.Timedelta(hours=1), ts2 + pd.Timedelta(hours=1)],
        }
    )
    all_raw = {"BTC/USDT": {"main": main, "htf": main}}
    idx = ld.build_timestamp_index(main)
    feed = PortfolioReplayFeed(clock, "1h")
    feed.prepare(
        all_raw=all_raw,
        all_main_index={"BTC/USDT": idx},
        test_timestamps=[ts0, ts1, ts2],
    )

    cfg = BacktestRunConfig(
        symbols=["BTC/USDT"],
        timeframe="1h",
        htf_timeframe="1h",
        initial_balance=1000.0,
        slippage=0.0003,
        taker_com=0.0004,
        leverage=1.0,
        risk_per_trade=0.01,
        reduced_risk_per_trade=0.01,
        reduce_risk_after_consecutive_losses=0,
        directional_proba_threshold=0.5,
        min_signal_gap=0.0,
        allow_longs=True,
        allow_shorts=True,
        max_new_positions_per_bar=1,
        max_open_positions=2,
        sl_cooldown_bars=0,
        max_sl_per_day=0,
        min_position_notional=10.0,
        realtime_features=False,
        charts_dir=Path("backtest_charts"),
        default_equity_chart_path=Path("backtest_charts/equity_curve.png"),
    )
    ctx = PortfolioBacktestContext(
        run_config=cfg,
        feature_names=[],
        event_filter_config={"enabled": False},
        symbol_categories=None,
        clip_bounds={},
        all_raw=all_raw,
        all_features={},
        all_features_prepared={},
        all_main_index={"BTC/USDT": idx},
        model=None,
        prediction_lookup=None,
    )

    executor = SimulationExecutor(clock=clock, slippage=cfg.slippage, taker_fee=cfg.taker_com)
    portfolio = PortfolioManager(cfg.initial_balance, taker_com=cfg.taker_com)
    risk = RiskManager(
        RiskLimits(
            risk_per_trade=cfg.risk_per_trade,
            reduced_risk_per_trade=cfg.reduced_risk_per_trade,
            reduce_risk_after_consecutive_losses=cfg.reduce_risk_after_consecutive_losses,
            leverage=cfg.leverage,
            min_position_notional=cfg.min_position_notional,
        )
    )
    execution = ExecutionService(
        BarEntryConstraints(
            max_new_positions_per_bar=cfg.max_new_positions_per_bar,
            max_open_positions=cfg.max_open_positions,
        )
    )
    brain = MagicMock(spec=SignalBrain)
    brain.predict.return_value = None

    engine = TradingEngine(
        clock=clock,
        data_feed=None,
        order_executor=executor,
        portfolio=portfolio,
        risk_manager=risk,
        execution_service=execution,
        signal_brain=brain,
        symbols=["BTC/USDT"],
        timeframe="1h",
        slippage=cfg.slippage,
        taker_fee=cfg.taker_com,
        portfolio_replay_feed=feed,
        portfolio_backtest_context=ctx,
    )

    engine.run()

    assert len(engine.equity_curve) == 3
    assert engine.portfolio.cash == cfg.initial_balance
    assert len(engine.trades) == 0
