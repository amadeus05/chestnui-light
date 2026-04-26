"""Тонкий CLI: запуск портфельного бэктеста (параметры из config.py)."""
from src.application.backtest import BacktestOrchestrator

if __name__ == "__main__":
    BacktestOrchestrator().run()
