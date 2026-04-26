"""Слой бэктеста: портфельный прогон (close t → exec t+1), данные и оркестратор отчётов."""

from src.application.backtest.backtest_config import BacktestRunConfig
from src.application.backtest.portfolio_backtest_context import PortfolioBacktestContext
from src.application.backtest.backtest_orchestrator import BacktestOrchestrator
from src.application.backtest.feature_providers import (
    DataFramePredictionProvider,
    PrecomputedFeatureProvider,
    RealtimeFeatureProvider,
)
from src.application.backtest import backtest_data
from src.application.backtest.portfolio_backtest_runner import run_portfolio_backtest
from src.application.feeds.portfolio_replay_feed import PortfolioReplayFeed

__all__ = [
    "BacktestOrchestrator",
    "BacktestRunConfig",
    "PortfolioBacktestContext",
    "PortfolioReplayFeed",
    "DataFramePredictionProvider",
    "PrecomputedFeatureProvider",
    "RealtimeFeatureProvider",
    "backtest_data",
    "run_portfolio_backtest",
]
