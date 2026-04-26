"""
Параллельная валидация: при RUN_BACKTEST_PARITY=1 можно добавить сравнение снимков trades/equity.

После делегирования backtest() в portfolio_backtest_runner полный дифф двух движков
заменяется регрессионными фикстурами на фиксированном периоде.
"""
from __future__ import annotations

import os

import pytest


def test_portfolio_runner_importable():
    from src.application.backtest.portfolio_backtest_runner import run_portfolio_backtest

    assert callable(run_portfolio_backtest)


@pytest.mark.skipif(
    os.environ.get("RUN_BACKTEST_PARITY") != "1",
    reason="Set RUN_BACKTEST_PARITY=1 и подготовьте market_data.db + модели для полного parity",
)
def test_full_parity_placeholder():
    pytest.skip("Добавьте сравнение equity/trades со снимком при готовности CI")
