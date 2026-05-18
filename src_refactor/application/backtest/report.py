from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src_refactor.application.backtest.metrics import BacktestBucketStats, BacktestMetrics


def format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def compact_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def format_float(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


@dataclass(frozen=True, slots=True)
class BacktestTextReportRenderer:
    title: str = "PORTFOLIO BACKTEST RESULTS"

    def render(self, metrics: BacktestMetrics) -> str:
        sections = [
            "Simulation finished.",
            "",
            self._render_header(),
            self._render_summary(metrics),
        ]
        if metrics.monthly:
            sections.extend(["", "Monthly Performance:", _render_table(
                ["Month", "PnL", "Total", "Wins", "Losses"],
                [
                    [
                        month,
                        format_signed_dollars(stats.pnl_abs),
                        str(stats.trades),
                        str(stats.wins),
                        str(stats.losses),
                    ]
                    for month, stats in metrics.monthly.items()
                ],
                right_align={1, 2, 3, 4},
            )])
        if metrics.by_symbol:
            sections.extend(["", "Summary by Coin:", _render_table(
                ["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"],
                [
                    [
                        compact_symbol(symbol),
                        str(stats.trades),
                        str(stats.tp),
                        str(stats.sl),
                        f"{stats.win_rate_pct:.1f}%",
                        format_signed_dollars(stats.pnl_abs),
                    ]
                    for symbol, stats in metrics.by_symbol.items()
                ],
                right_align={1, 2, 3, 4, 5},
            )])
        sections.extend(["", "Long / Short Summary:", _render_table(
            ["Direction", "Total", "Wins", "Losses", "PnL"],
            [
                [
                    direction,
                    str(stats.trades),
                    str(stats.wins),
                    str(stats.losses),
                    format_signed_dollars(stats.pnl_abs),
                ]
                for direction, stats in metrics.by_direction.items()
            ],
            right_align={1, 2, 3, 4},
        )])
        return "\n".join(sections)

    def _render_header(self) -> str:
        title = self.title[:72]
        return f"=== {title} ==="

    @staticmethod
    def _render_summary(metrics: BacktestMetrics) -> str:
        summary = metrics.summary
        return "\n".join(
            [
                f"Trades: {summary.total_trades} (W: {summary.total_wins} / L: {summary.total_losses})",
                "Equity:",
                f"  Start: ${summary.initial_balance:.2f}",
                f"  End:   ${summary.final_balance:.2f}",
                (
                    "  PnL:   "
                    f"{format_signed_dollars(summary.total_pnl_abs)} "
                    f"({format_percent_value(summary.total_return_pct)})"
                ),
                f"  Fees:  ${summary.total_fees:.2f}",
                "Risk:",
                f"  Max DD: {summary.max_drawdown_pct:.2f}%",
                f"  Profit Factor: {format_float(summary.profit_factor)}",
                f"  Expectancy: {format_signed_dollars(summary.expectancy)}",
                f"  Sharpe: {summary.sharpe:.2f}",
            ]
        )


@dataclass(frozen=True, slots=True)
class BacktestEquityCurveRenderer:
    dpi: int = 150

    def save(self, metrics: BacktestMetrics, path: Path | str) -> Path | None:
        if len(metrics.equity_curve) <= 1:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timestamps = [timestamp for timestamp, _ in metrics.equity_curve]
        equity = [value for _, value in metrics.equity_curve]
        summary = metrics.summary
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity, label="Portfolio Equity")
        plt.axhline(y=summary.initial_balance, linestyle="--")
        plt.title(
            "Multi-Symbol Equity Curve | "
            f"{summary.total_trades} trades | DD: {summary.max_drawdown_pct:.1f}%"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        return output_path


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right_align: set[int] | None = None,
) -> str:
    right_align = right_align or set()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_row(row_values: list[str]) -> str:
        formatted = []
        for index, cell in enumerate(row_values):
            text = str(cell)
            formatted.append(text.rjust(widths[index]) if index in right_align else text.ljust(widths[index]))
        return "| " + " | ".join(formatted) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])
