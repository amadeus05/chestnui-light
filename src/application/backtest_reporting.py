"""Консольная отчётность и график equity по итогам бэктеста."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compact_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
    right_align = right_align or set()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(row_values: list[str]) -> str:
        formatted = []
        for idx, cell in enumerate(row_values):
            text = str(cell)
            if idx in right_align:
                formatted.append(text.rjust(widths[idx]))
            else:
                formatted.append(text.ljust(widths[idx]))
        return "│ " + " │ ".join(formatted) + " │"

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    print(top)
    print(format_row(headers))
    print(mid)
    for row in rows:
        print(format_row(row))
    print(bottom)


@dataclass(frozen=True)
class BacktestMetrics:
    sharpe: float
    sortino: float
    calmar: float
    cagr: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    total_wins: int
    total_losses: int
    total_pnl_abs: float
    total_fees: float
    total_return_pct: float
    expectancy: float
    final_balance: float


def compute_backtest_metrics(
    *,
    equity_curve: list[float],
    equity_timestamps: list,
    trades: list[dict],
    max_drawdown: float,
    final_balance: float,
    initial_balance: float,
) -> BacktestMetrics:
    sharpe = sortino = calmar = cagr = 0.0
    pf = 0.0

    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve, index=equity_timestamps)
        daily_equity = equity_series.resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
            cagr = (
                ((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1) if total_days > 0 else 0
            )
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe = (mean_daily_return / std_daily_return) * np.sqrt(365)

            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = (mean_daily_return / downside_returns.std()) * np.sqrt(365)
            else:
                sortino = 0.0

            calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0.0

        if trades:
            returns = np.array([t["pnl_abs"] for t in trades])
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_trades = len(trades)
    total_wins = sum(1 for trade in trades if trade["pnl_abs"] > 0)
    total_losses = total_trades - total_wins
    total_pnl_abs = sum(trade["pnl_abs"] for trade in trades)
    total_fees = sum(trade.get("commission", 0.0) for trade in trades)
    total_return_pct = ((final_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
    expectancy = (total_pnl_abs / total_trades) if total_trades > 0 else 0.0

    return BacktestMetrics(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        cagr=cagr,
        profit_factor=pf,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        total_wins=total_wins,
        total_losses=total_losses,
        total_pnl_abs=total_pnl_abs,
        total_fees=total_fees,
        total_return_pct=total_return_pct,
        expectancy=expectancy,
        final_balance=final_balance,
    )


class BacktestReporter:
    """Печать итоговых таблиц и сохранение графика equity."""

    def __init__(
        self,
        *,
        result_title: str,
        default_equity_chart_path: Path,
        equity_curve_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        self.result_title = result_title
        self.default_equity_chart_path = default_equity_chart_path
        self.equity_curve_path = equity_curve_path
        self.show_plot = show_plot

    def print_summary(self, metrics: BacktestMetrics, *, initial_balance: float) -> None:
        print("\nSimulation finished.\n")
        print("╔═══════════════════════════════════════════════════════════╗")
        print(f"║{self.result_title[:59].center(59)}║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(f"\n📊 Trades: {metrics.total_trades} (W: {metrics.total_wins} / L: {metrics.total_losses})")
        print("💰 Equity:")
        print(f"   Start: ${initial_balance:.2f}")
        print(f"   End:   ${metrics.final_balance:.2f}")
        print(
            f"   PnL:   {format_signed_dollars(metrics.total_pnl_abs)} "
            f"({format_percent_value(metrics.total_return_pct)})"
        )
        print(f"   Fees:  ${metrics.total_fees:.2f}")
        print("📉 Risk:")
        print(f"   Max DD: {metrics.max_drawdown:.2f}%")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        print(f"   Expectancy: {format_signed_dollars(metrics.expectancy)}")
        print(f"   Sharpe: {metrics.sharpe:.2f}")

    def print_monthly_table(self, monthly_stats: dict[str, dict]) -> None:
        monthly_rows = []
        for month_key in sorted(monthly_stats.keys()):
            stats = monthly_stats[month_key]
            monthly_rows.append(
                [
                    month_key,
                    format_signed_dollars(stats["pnl_abs"]),
                    str(stats["trades"]),
                    str(stats["wins"]),
                    str(stats["losses"]),
                ]
            )

        if monthly_rows:
            print("\n📅 Monthly Performance Extended:")
            print_table(
                ["Month", "PnL", "Total", "Wins", "Losses"],
                monthly_rows,
                right_align={1, 2, 3, 4},
            )

    def print_symbol_breakdown(self, trades: list[dict]) -> None:
        symbol_stats: dict[str, dict] = {}
        for trade in trades:
            stats = symbol_stats.setdefault(
                trade["sym"],
                {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
            )
            stats["trades"] += 1
            stats["pnl_abs"] += trade["pnl_abs"]
            if trade["reason"] == "TP":
                stats["tp"] += 1
            if trade["reason"] == "SL":
                stats["sl"] += 1
            if trade["pnl_abs"] > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        if not symbol_stats:
            return

        coin_rows = []
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            winrate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0.0
            coin_rows.append(
                [
                    compact_symbol(symbol),
                    str(stats["trades"]),
                    str(stats["tp"]),
                    str(stats["sl"]),
                    f"{winrate:.1f}%",
                    format_signed_dollars(stats["pnl_abs"]),
                ]
            )

        print("\n📊 Summary by Coin:")
        print_table(
            ["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"],
            coin_rows,
            right_align={1, 2, 3, 4, 5},
        )

    def print_direction_breakdown(self, trades: list[dict]) -> None:
        direction_stats = {
            "LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
            "SHORT": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        }
        for trade in trades:
            stats = direction_stats[trade["direction"]]
            stats["total"] += 1
            stats["pnl_abs"] += trade["pnl_abs"]
            if trade["pnl_abs"] > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        direction_rows = []
        for direction in ("LONG", "SHORT"):
            stats = direction_stats[direction]
            direction_rows.append(
                [
                    direction,
                    str(stats["total"]),
                    str(stats["wins"]),
                    str(stats["losses"]),
                    format_signed_dollars(stats["pnl_abs"]),
                ]
            )

        print("\n📈 Long / Short Summary:")
        print_table(
            ["Direction", "Total", "Wins", "Losses", "PnL"],
            direction_rows,
            right_align={1, 2, 3, 4},
        )

    def print_full_console(
        self,
        metrics: BacktestMetrics,
        *,
        initial_balance: float,
        monthly_stats: dict[str, dict],
        trades: list[dict],
    ) -> None:
        self.print_summary(metrics, initial_balance=initial_balance)
        self.print_monthly_table(monthly_stats)
        self.print_symbol_breakdown(trades)
        self.print_direction_breakdown(trades)

    def save_equity_chart(
        self,
        *,
        equity_timestamps: list,
        equity_curve: list[float],
        initial_balance: float,
        total_trades: int,
        max_drawdown: float,
    ) -> Path | None:
        if len(equity_curve) <= 1:
            return None

        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, label="Portfolio Equity")
        plt.axhline(y=initial_balance, linestyle="--")
        plt.title(f"Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        output_chart_path = Path(self.equity_curve_path or self.default_equity_chart_path)
        output_chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_chart_path, dpi=150)
        if self.show_plot:
            plt.show()
        else:
            plt.close()
        print(f"\nSaved chart: {output_chart_path}")
        return output_chart_path
