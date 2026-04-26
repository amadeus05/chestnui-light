"""Единственный оффлайн-бэктест: сигнал на закрытии бара t → исполнение на баре t+1.

Реализация: ``BacktestOrchestrator`` → ``run_portfolio_backtest`` (PortfolioManager,
bar_fill, SignalBrain, отчётность). Это не ``TradingEngine`` + ``ReplayDataFeed`` —
тот путь другой порядок решений по барам и не используется как основной бэктест.
"""
from __future__ import annotations

import sys

from src.application.backtest import BacktestOrchestrator


def main() -> int:
    print("=" * 60)
    print("BACKTEST (канонический)")
    print("  Семантика: решение на close(t) → исполнение на баре t+1")
    print("=" * 60)

    try:
        metrics = BacktestOrchestrator().run()
        if metrics is None:
            return 1
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
