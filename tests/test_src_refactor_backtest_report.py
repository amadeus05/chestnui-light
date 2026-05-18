from pathlib import Path

import pandas as pd

from src_refactor.application.backtest import (
    BacktestEquityCurveRenderer,
    BacktestMetrics,
    BacktestSummaryMetrics,
    BacktestTextReportRenderer,
)
from src_refactor.application.backtest.metrics import BacktestBucketStats


def _metrics() -> BacktestMetrics:
    return BacktestMetrics(
        summary=BacktestSummaryMetrics(
            initial_balance=100.0,
            final_balance=103.0,
            total_trades=2,
            total_wins=1,
            total_losses=1,
            win_rate_pct=50.0,
            total_pnl_abs=3.0,
            total_fees=0.5,
            total_return_pct=3.0,
            expectancy=1.5,
            max_drawdown_pct=5.0,
            profit_factor=2.5,
            sharpe=1.25,
            sortino=0.75,
            calmar=0.5,
            cagr=0.025,
        ),
        monthly={
            "2025-01": BacktestBucketStats(trades=2, wins=1, losses=1, pnl_abs=3.0, tp=1, sl=1),
        },
        by_symbol={
            "BTC/USDT": BacktestBucketStats(trades=2, wins=1, losses=1, pnl_abs=3.0, tp=1, sl=1),
        },
        by_direction={
            "LONG": BacktestBucketStats(trades=1, wins=1, losses=0, pnl_abs=5.0, tp=1, sl=0),
            "SHORT": BacktestBucketStats(trades=1, wins=0, losses=1, pnl_abs=-2.0, tp=0, sl=1),
        },
        equity_curve=(
            (pd.Timestamp("2025-01-01"), 100.0),
            (pd.Timestamp("2025-01-02"), 103.0),
        ),
    )


def test_backtest_text_report_renders_legacy_summary_sections():
    report = BacktestTextReportRenderer(title="WALK-FORWARD OOS BACKTEST").render(_metrics())

    assert "=== WALK-FORWARD OOS BACKTEST ===" in report
    assert "Trades: 2 (W: 1 / L: 1)" in report
    assert "PnL:   +$3.00 (+3.00%)" in report
    assert "Monthly Performance:" in report
    assert "Summary by Coin:" in report
    assert "BTCUSDT" in report
    assert "Long / Short Summary:" in report


def test_backtest_equity_curve_renderer_saves_chart():
    output_path = Path("src_refactor/.tmp_tests/equity_curve_report_test.png")
    try:
        saved_path = BacktestEquityCurveRenderer().save(_metrics(), output_path)

        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        output_path.unlink(missing_ok=True)
