"""Оркестратор: единая точка запуска бэктеста и отчётности (ядро — portfolio_backtest_runner)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.application.backtest.backtest_config import BacktestRunConfig
from src.application.backtest.portfolio_backtest_runner import run_portfolio_backtest
from src.application.backtest_reporting import BacktestMetrics


@dataclass
class BacktestOrchestrator:
    """Тонкая обёртка над run_portfolio_backtest (семантика close(t) → exec(t+1))."""

    run_config: BacktestRunConfig | None = None

    def run(
        self,
        *,
        model_name: str | None = None,
        model=None,
        features_meta: dict | None = None,
        predictions: pd.DataFrame | None = None,
        equity_curve_path: Path | None = None,
        result_title: str = "PORTFOLIO BACKTEST RESULTS",
    ) -> BacktestMetrics | None:
        return run_portfolio_backtest(
            model_name=model_name,
            model=model,
            features_meta=features_meta,
            predictions=predictions,
            equity_curve_path=equity_curve_path,
            result_title=result_title,
            run_config=self.run_config,
        )
