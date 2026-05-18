import math

import pytest
import pandas as pd

from src_refactor.application.backtest import BacktestMetricsCalculator
from src_refactor.domain.portfolio.portfolio_manager import PortfolioState, Trade


def _trade(
    *,
    trade_number: int,
    symbol: str,
    direction: int,
    reason: str,
    pnl_abs: float,
    commission: float = 0.1,
    closed_at: str = "2025-01-01 01:00:00",
) -> Trade:
    return Trade(
        trade_number=trade_number,
        symbol=symbol,
        direction=direction,
        reason=reason,
        opened_at=pd.Timestamp("2025-01-01 00:00:00"),
        closed_at=pd.Timestamp(closed_at),
        entry_price=100.0,
        exit_price=101.0,
        pnl_pct=pnl_abs / 100.0,
        pnl_abs=pnl_abs,
        commission=commission,
    )


def test_backtest_metrics_match_legacy_summary_formulas():
    state = PortfolioState(
        balance=103.0,
        max_drawdown_pct=5.0,
        trades=[
            _trade(trade_number=1, symbol="BTC/USDT", direction=1, reason="TP", pnl_abs=5.0, commission=0.2),
            _trade(trade_number=2, symbol="ETH/USDT", direction=-1, reason="SL", pnl_abs=-2.0, commission=0.3),
        ],
        equity_curve=[
            (pd.Timestamp("2025-01-01"), 100.0),
            (pd.Timestamp("2025-01-02"), 105.0),
            (pd.Timestamp("2025-01-03"), 103.0),
        ],
    )

    metrics = BacktestMetricsCalculator.from_portfolio_state(state, initial_balance=100.0)
    summary = metrics.summary

    assert summary.total_trades == 2
    assert summary.total_wins == 1
    assert summary.total_losses == 1
    assert summary.win_rate_pct == pytest.approx(50.0)
    assert summary.total_pnl_abs == pytest.approx(3.0)
    assert summary.total_fees == pytest.approx(0.5)
    assert summary.total_return_pct == pytest.approx(3.0)
    assert summary.expectancy == pytest.approx(1.5)
    assert summary.profit_factor == pytest.approx(2.5)
    assert summary.max_drawdown_pct == pytest.approx(5.0)
    assert math.isfinite(summary.sharpe)


def test_backtest_metrics_build_month_symbol_and_direction_buckets():
    state = PortfolioState(
        balance=103.0,
        trades=[
            _trade(trade_number=1, symbol="BTC/USDT", direction=1, reason="TP", pnl_abs=5.0),
            _trade(
                trade_number=2,
                symbol="BTC/USDT",
                direction=-1,
                reason="SL",
                pnl_abs=-2.0,
                closed_at="2025-02-01 01:00:00",
            ),
        ],
    )

    metrics = BacktestMetricsCalculator.from_portfolio_state(state, initial_balance=100.0)

    assert metrics.monthly["2025-01"].trades == 1
    assert metrics.monthly["2025-02"].losses == 1
    assert metrics.by_symbol["BTC/USDT"].tp == 1
    assert metrics.by_symbol["BTC/USDT"].sl == 1
    assert metrics.by_direction["LONG"].wins == 1
    assert metrics.by_direction["SHORT"].losses == 1
