from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src_refactor.domain.portfolio.portfolio_manager import PortfolioState, Trade


@dataclass(frozen=True, slots=True)
class BacktestSummaryMetrics:
    initial_balance: float
    final_balance: float
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate_pct: float
    total_pnl_abs: float
    total_fees: float
    total_return_pct: float
    expectancy: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe: float
    sortino: float
    calmar: float
    cagr: float


@dataclass(frozen=True, slots=True)
class BacktestBucketStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_abs: float = 0.0
    tp: int = 0
    sl: int = 0

    @property
    def win_rate_pct(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades > 0 else 0.0


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    summary: BacktestSummaryMetrics
    monthly: dict[str, BacktestBucketStats] = field(default_factory=dict)
    by_symbol: dict[str, BacktestBucketStats] = field(default_factory=dict)
    by_direction: dict[str, BacktestBucketStats] = field(default_factory=dict)
    equity_curve: tuple[tuple[pd.Timestamp, float], ...] = ()


class BacktestMetricsCalculator:
    @staticmethod
    def from_portfolio_state(state: PortfolioState, *, initial_balance: float) -> BacktestMetrics:
        trades = tuple(state.trades)
        equity_curve = tuple(
            (pd.to_datetime(timestamp), float(equity))
            for timestamp, equity in state.equity_curve
        )
        summary = _summary_metrics(
            trades,
            equity_curve,
            initial_balance=float(initial_balance),
            final_balance=float(state.balance),
            max_drawdown_pct=float(state.max_drawdown_pct),
        )
        return BacktestMetrics(
            summary=summary,
            monthly=_bucket_by_month(trades),
            by_symbol=_bucket_by_symbol(trades),
            by_direction=_bucket_by_direction(trades),
            equity_curve=equity_curve,
        )


def _summary_metrics(
    trades: tuple[Trade, ...],
    equity_curve: tuple[tuple[pd.Timestamp, float], ...],
    *,
    initial_balance: float,
    final_balance: float,
    max_drawdown_pct: float,
) -> BacktestSummaryMetrics:
    total_trades = len(trades)
    total_wins = sum(1 for trade in trades if trade.pnl_abs > 0)
    total_losses = total_trades - total_wins
    total_pnl_abs = sum(float(trade.pnl_abs) for trade in trades)
    total_fees = sum(float(trade.commission) for trade in trades)
    win_rate_pct = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    total_return_pct = ((final_balance - initial_balance) / initial_balance * 100.0) if initial_balance > 0 else 0.0
    expectancy = (total_pnl_abs / total_trades) if total_trades > 0 else 0.0
    profit_factor = _profit_factor(trades)
    risk_adjusted = _risk_adjusted_metrics(equity_curve, max_drawdown_pct)
    return BacktestSummaryMetrics(
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_trades=total_trades,
        total_wins=total_wins,
        total_losses=total_losses,
        win_rate_pct=win_rate_pct,
        total_pnl_abs=total_pnl_abs,
        total_fees=total_fees,
        total_return_pct=total_return_pct,
        expectancy=expectancy,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        sharpe=risk_adjusted["sharpe"],
        sortino=risk_adjusted["sortino"],
        calmar=risk_adjusted["calmar"],
        cagr=risk_adjusted["cagr"],
    )


def _profit_factor(trades: tuple[Trade, ...]) -> float:
    if not trades:
        return 0.0
    returns = np.array([trade.pnl_abs for trade in trades], dtype=float)
    gross_profit = sum(value for value in returns if value > 0)
    gross_loss = abs(sum(value for value in returns if value < 0))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _risk_adjusted_metrics(
    equity_curve: tuple[tuple[pd.Timestamp, float], ...],
    max_drawdown_pct: float,
) -> dict[str, float]:
    if not equity_curve:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "cagr": 0.0}

    equity_series = pd.Series(
        [equity for _, equity in equity_curve],
        index=[timestamp for timestamp, _ in equity_curve],
        dtype=float,
    )
    daily_equity = equity_series.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    if len(daily_returns) <= 1 or daily_returns.std() <= 0:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "cagr": 0.0}

    total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
    cagr = ((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1) if total_days > 0 else 0.0
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe = (mean_daily_return / std_daily_return) * np.sqrt(365)
    downside_returns = daily_returns[daily_returns < 0]
    sortino = (
        (mean_daily_return / downside_returns.std()) * np.sqrt(365)
        if len(downside_returns) > 1 and downside_returns.std() > 0
        else 0.0
    )
    calmar = cagr / (max_drawdown_pct / 100.0) if max_drawdown_pct > 0 else 0.0
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "cagr": float(cagr),
    }


def _bucket_by_month(trades: tuple[Trade, ...]) -> dict[str, BacktestBucketStats]:
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(pd.to_datetime(trade.closed_at).strftime("%Y-%m"), []).append(trade)
    return {key: _bucket_stats(tuple(value)) for key, value in sorted(buckets.items())}


def _bucket_by_symbol(trades: tuple[Trade, ...]) -> dict[str, BacktestBucketStats]:
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(trade.symbol, []).append(trade)
    return {key: _bucket_stats(tuple(value)) for key, value in sorted(buckets.items())}


def _bucket_by_direction(trades: tuple[Trade, ...]) -> dict[str, BacktestBucketStats]:
    buckets = {"LONG": [], "SHORT": []}
    for trade in trades:
        buckets["LONG" if trade.direction == 1 else "SHORT"].append(trade)
    return {key: _bucket_stats(tuple(value)) for key, value in buckets.items()}


def _bucket_stats(trades: tuple[Trade, ...]) -> BacktestBucketStats:
    return BacktestBucketStats(
        trades=len(trades),
        wins=sum(1 for trade in trades if trade.pnl_abs > 0),
        losses=sum(1 for trade in trades if trade.pnl_abs <= 0),
        pnl_abs=sum(float(trade.pnl_abs) for trade in trades),
        tp=sum(1 for trade in trades if trade.reason == "TP"),
        sl=sum(1 for trade in trades if trade.reason == "SL"),
    )
