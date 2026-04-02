"""
Веб-дашборд для пейпер-трейдинга.
Запуск: python paper_web.py
Открыть: http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config as cfg
from etl import create_exchange_service
from src.persistence.repositories.base_trades_repository import BaseTradesRepository
from src.persistence.repositories.trades_repository_factory import create_trades_repository_from_config

logger = logging.getLogger("paper_web")

try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    raise ImportError("Установите Flask: pip install flask")

app = Flask(__name__)

EXEC_TYPE = "paper"
EXIT_REASONS_CACHE_TTL_SEC = 60.0

_repo_lock = threading.Lock()
_repo_instance: BaseTradesRepository | None = None
_repo_initialized = False
_repo_signature: tuple[str, str | None, str | None, str | None] | None = None
_exit_reasons_cache: tuple[float, list[str]] | None = None


def _get_repo_signature() -> tuple[str, str | None, str | None, str | None]:
    """Собрать сигнатуру текущей конфигурации backend-репозитория."""
    db_type = str(getattr(cfg, "EXECUTION_DB_TYPE", "sqlite")).lower()
    if db_type == "sqlite":
        db_path = str(getattr(cfg, "EXECUTION_DB_PATH", None) or getattr(cfg, "DB_PATH", ""))
        return (db_type, db_path, None, None)

    supabase_url = getattr(cfg, "SUPABASE_URL", None)
    supabase_key = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or getattr(cfg, "SUPABASE_KEY", None)
    )
    return (db_type, None, str(supabase_url) if supabase_url else None, supabase_key)


def _invalidate_repo_cache() -> None:
    """Сбросить кэш репозитория и производных данных."""
    global _repo_instance, _repo_initialized, _repo_signature, _exit_reasons_cache
    _repo_instance = None
    _repo_initialized = False
    _repo_signature = None
    _exit_reasons_cache = None


def _get_repo() -> BaseTradesRepository:
    """Вернуть общий репозиторий, инициализируя схему только один раз."""
    global _repo_instance, _repo_initialized, _repo_signature
    current_signature = _get_repo_signature()

    if (
        _repo_instance is not None
        and _repo_initialized
        and _repo_signature == current_signature
    ):
        return _repo_instance

    with _repo_lock:
        if _repo_signature != current_signature:
            _invalidate_repo_cache()
            _repo_signature = current_signature
        if _repo_instance is None:
            _repo_instance = create_trades_repository_from_config()
        if not _repo_initialized:
            _repo_instance.init_schema()
            _repo_initialized = True
        return _repo_instance


def _get_available_symbols() -> list[str]:
    """Получить список доступных символов из конфига."""
    return [str(s) for s in getattr(cfg, "SYMBOLS", [])]


def _is_supabase_repo(repo: BaseTradesRepository) -> bool:
    return hasattr(repo, "client")


def _fetch_closed_trades(
    repo: BaseTradesRepository,
    columns: list[str],
    *,
    date_from_ms: int | None = None,
    date_to_ms: int | None = None,
    symbols: list[str] | None = None,
    direction: str | None = None,
    exit_reasons: list[str] | None = None,
    order_desc: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Унифицированная выборка закрытых сделок для SQLite/Supabase."""
    columns_sql = ", ".join(columns)

    if _is_supabase_repo(repo):
        query = (
            repo.client.table("execution_trades")
            .select(columns_sql)
            .eq("execution_type", EXEC_TYPE)
            .eq("status", "closed")
        )
        if date_from_ms:
            query = query.gte("exit_ts_ms", date_from_ms)
        if date_to_ms:
            query = query.lt("exit_ts_ms", date_to_ms + 24 * 60 * 60 * 1000)
        if symbols:
            query = query.in_("symbol", symbols)
        if direction:
            dir_map = {"long": 1, "short": -1}
            d = dir_map.get(direction.lower())
            if d is not None:
                query = query.eq("direction", d)
        if exit_reasons:
            query = query.in_("exit_reason", exit_reasons)

        query = query.order("exit_ts_ms", desc=order_desc)
        if limit is not None:
            query = query.limit(limit)
        response = query.execute()
        return list(response.data or [])

    where_clause, params = build_trades_query(
        date_from_ms, date_to_ms, symbols, direction, exit_reasons
    )
    order_sql = "DESC" if order_desc else "ASC"
    sql = (
        f"SELECT {columns_sql} "
        f"FROM execution_trades "
        f"WHERE {where_clause} "
        f"ORDER BY exit_ts_ms {order_sql}"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params = params + [limit]

    conn = sqlite3.connect(repo.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _get_exit_reasons(repo: BaseTradesRepository) -> list[str]:
    """Получить уникальные причины выхода из БД."""
    global _exit_reasons_cache
    default_reasons = ["TP", "SL", "timeout", "manual"]
    now = time.time()
    if _exit_reasons_cache is not None:
        cached_at, cached_values = _exit_reasons_cache
        if now - cached_at < EXIT_REASONS_CACHE_TTL_SEC:
            return cached_values
    try:
        rows = _fetch_closed_trades(
            repo,
            columns=["exit_reason"],
            order_desc=True,
            limit=1000,
        )
        reasons = sorted({str(r.get("exit_reason")) for r in rows if r.get("exit_reason")})
        result = reasons or default_reasons
        _exit_reasons_cache = (now, result)
        return result
    except Exception:
        return default_reasons


@dataclass
class DashboardMetrics:
    """Метрики дашборда."""

    exchange: str
    timeframe: str
    htf_timeframe: str
    leverage: float
    initial_balance: float
    wallet_balance: float
    realized_pnl: float
    open_margin: float
    available_balance: float
    open_positions_count: int
    closed_trades_count: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown_pct: float
    best_trade: float
    worst_trade: float
    total_trades: int
    longs_count: int
    shorts_count: int
    # Фильтрованные метрики
    filtered_realized_pnl: float = 0.0
    filtered_win_rate: float = 0.0
    filtered_trades_count: int = 0


def _db_path() -> str:
    p = getattr(cfg, "EXECUTION_DB_PATH", None)
    if p:
        return str(p)
    return str(cfg.DB_PATH)


def _get_leverage() -> float:
    return float(getattr(cfg, "LEVERAGE", 1))


def _get_initial_balance() -> float:
    return float(
        getattr(
            cfg, "PAPER_INITIAL_BALANCE", getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)
        )
    )


def _parse_date(date_str: str | None) -> int | None:
    """Парсит строку даты в timestamp ms."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def build_trades_query(
    date_from_ms: int | None,
    date_to_ms: int | None,
    symbols: list[str] | None,
    direction: str | None,
    exit_reasons: list[str] | None,
) -> tuple[str, list]:
    """Построить SQL запрос с фильтрами."""
    conditions = ["execution_type = ?", "status = 'closed'"]
    params: list = [EXEC_TYPE]

    if date_from_ms:
        conditions.append("exit_ts_ms >= ?")
        params.append(date_from_ms)

    if date_to_ms:
        # Добавляем время до конца дня
        conditions.append("exit_ts_ms < ?")
        params.append(date_to_ms + 24 * 60 * 60 * 1000)

    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        conditions.append(f"symbol IN ({placeholders})")
        params.extend(symbols)

    if direction:
        dir_map = {"long": 1, "short": -1}
        if direction.lower() in dir_map:
            conditions.append("direction = ?")
            params.append(dir_map[direction.lower()])

    if exit_reasons:
        placeholders = ",".join(["?"] * len(exit_reasons))
        conditions.append(f"exit_reason IN ({placeholders})")
        params.extend(exit_reasons)

    where_clause = " AND ".join(conditions)
    return where_clause, params


def calculate_metrics(
    repo: BaseTradesRepository,
    date_from_ms: int | None = None,
    date_to_ms: int | None = None,
    symbols: list[str] | None = None,
    direction: str | None = None,
    exit_reasons: list[str] | None = None,
    open_trades: list | None = None,
    all_closed_trades: list[dict] | None = None,
    filtered_closed_trades: list[dict] | None = None,
) -> DashboardMetrics:
    """Рассчитать все метрики для дашборда с учетом фильтров."""
    leverage = _get_leverage()
    initial = _get_initial_balance()
    realized_pnl = repo.sum_closed_pnl_quote(EXEC_TYPE)
    wallet = initial + realized_pnl
    open_margin = repo.sum_open_margin_quote(EXEC_TYPE, leverage)
    available = wallet - open_margin

    if open_trades is None:
        open_trades = repo.list_open_trades(EXEC_TYPE)
    open_count = len(open_trades)

    # Получаем закрытые сделки для статистики (без фильтров для общих метрик)
    if all_closed_trades is None:
        all_closed_trades = _fetch_closed_trades(
            repo,
            columns=["direction", "pnl_quote", "pnl_pct", "exit_reason"],
            order_desc=False,
        )

    # Получаем отфильтрованные сделки
    if filtered_closed_trades is None:
        has_active_filters = bool(date_from_ms or date_to_ms or symbols or direction or exit_reasons)
        if has_active_filters:
            filtered_closed_trades = _fetch_closed_trades(
                repo,
                columns=["direction", "pnl_quote", "pnl_pct", "exit_reason"],
                date_from_ms=date_from_ms,
                date_to_ms=date_to_ms,
                symbols=symbols,
                direction=direction,
                exit_reasons=exit_reasons,
                order_desc=False,
            )
        else:
            filtered_closed_trades = all_closed_trades

    closed_count = len(all_closed_trades)
    total_trades = open_count + closed_count

    # Расчет общих метрик
    wins = [t for t in all_closed_trades if t["pnl_quote"] > 0]
    losses = [t for t in all_closed_trades if t["pnl_quote"] <= 0]

    win_count = len(wins)
    loss_count = len(losses)

    win_rate = (win_count / closed_count * 100) if closed_count > 0 else 0.0

    total_wins = sum(t["pnl_quote"] for t in wins)
    total_losses = abs(sum(t["pnl_quote"] for t in losses))
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

    avg_win = (total_wins / win_count) if win_count > 0 else 0.0
    avg_loss = (total_losses / loss_count) if loss_count > 0 else 0.0

    best_trade = max((t["pnl_quote"] for t in all_closed_trades), default=0.0)
    worst_trade = min((t["pnl_quote"] for t in all_closed_trades), default=0.0)

    # Расчет максимальной просадки
    max_dd = _calculate_max_drawdown(all_closed_trades, initial)

    longs = sum(1 for t in open_trades if t.direction == 1)
    shorts = sum(1 for t in open_trades if t.direction == -1)

    # Расчет фильтрованных метрик
    filtered_wins = [t for t in filtered_closed_trades if t["pnl_quote"] > 0]
    filtered_losses = [t for t in filtered_closed_trades if t["pnl_quote"] <= 0]
    filtered_win_count = len(filtered_wins)
    filtered_total = len(filtered_closed_trades)
    filtered_win_rate = (filtered_win_count / filtered_total * 100) if filtered_total > 0 else 0.0
    filtered_pnl = sum(t["pnl_quote"] for t in filtered_closed_trades)

    return DashboardMetrics(
        exchange=str(getattr(cfg, "ACTIVE_EXCHANGE", "unknown")).upper(),
        timeframe=str(getattr(cfg, "TIMEFRAME", "1h")),
        htf_timeframe=str(getattr(cfg, "HTF_TIMEFRAME", "4h")),
        leverage=leverage,
        initial_balance=initial,
        wallet_balance=wallet,
        realized_pnl=realized_pnl,
        open_margin=open_margin,
        available_balance=available,
        open_positions_count=open_count,
        closed_trades_count=closed_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown_pct=max_dd,
        best_trade=best_trade,
        worst_trade=worst_trade,
        total_trades=total_trades,
        longs_count=longs,
        shorts_count=shorts,
        filtered_realized_pnl=filtered_pnl,
        filtered_win_rate=filtered_win_rate,
        filtered_trades_count=filtered_total,
    )


def _calculate_max_drawdown(trades: list, initial_balance: float) -> float:
    """Рассчитать максимальную просадку в процентах."""
    if not trades:
        return 0.0

    peak = initial_balance
    max_dd = 0.0
    current_balance = initial_balance

    for trade in trades:
        current_balance += trade["pnl_quote"]
        if current_balance > peak:
            peak = current_balance
        if peak > 0:
            dd = (peak - current_balance) / peak * 100
            max_dd = max(max_dd, dd)

    return max_dd


def get_open_positions(
    repo: BaseTradesRepository,
    trades: list | None = None,
) -> list[dict]:
    """Получить список открытых позиций."""
    if trades is None:
        trades = repo.list_open_trades(EXEC_TYPE)
    positions = []
    for t in trades:
        p_long = getattr(t, "p_long", None)
        p_short = getattr(t, "p_short", None)
        positions.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 8),
                "entry_notional": round(t.entry_notional, 2),
                "stop_pct": round(t.stop_pct * 100, 2),
                "take_pct": round(t.take_pct * 100, 2),
                "p_long": round(float(p_long), 3) if p_long is not None else None,
                "p_short": round(float(p_short), 3) if p_short is not None else None,
                "entry_time": "-",
            }
        )
    return positions


def get_filtered_trades(
    repo: BaseTradesRepository,
    date_from_ms: int | None = None,
    date_to_ms: int | None = None,
    symbols: list[str] | None = None,
    direction: str | None = None,
    exit_reasons: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Получить отфильтрованные закрытые сделки."""
    rows = _fetch_closed_trades(
        repo,
        columns=[
            "symbol",
            "direction",
            "entry_price",
            "exit_price",
            "entry_notional",
            "pnl_quote",
            "pnl_pct",
            "exit_reason",
            "exit_ts_ms",
        ],
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols,
        direction=direction,
        exit_reasons=exit_reasons,
        order_desc=True,
        limit=limit,
    )
    trades = []
    for row in rows:
        trades.append(
            {
                "symbol": row["symbol"],
                "direction": "LONG" if row["direction"] == 1 else "SHORT",
                "entry_price": round(row["entry_price"], 8),
                "exit_price": round(row["exit_price"], 8),
                "notional": round(row["entry_notional"], 2),
                "pnl": round(row["pnl_quote"], 2),
                "pnl_pct": round(row["pnl_pct"], 2),
                "exit_reason": row["exit_reason"],
                "exit_time": datetime.fromtimestamp(
                    row["exit_ts_ms"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
                "is_win": row["pnl_quote"] > 0,
            }
        )
    return trades


def _format_recent_trades(rows: list[dict], limit: int = 100) -> list[dict]:
    """Преобразовать строки сделок в формат для UI, начиная с самых свежих."""
    trades: list[dict] = []
    for row in reversed(rows[-limit:]):
        trades.append(
            {
                "symbol": row["symbol"],
                "direction": "LONG" if row["direction"] == 1 else "SHORT",
                "entry_price": round(row["entry_price"], 8),
                "exit_price": round(row["exit_price"], 8),
                "notional": round(row["entry_notional"], 2),
                "pnl": round(row["pnl_quote"], 2),
                "pnl_pct": round(row["pnl_pct"], 2),
                "exit_reason": row["exit_reason"],
                "exit_time": datetime.fromtimestamp(
                    row["exit_ts_ms"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
                "is_win": row["pnl_quote"] > 0,
            }
        )
    return trades


def get_equity_curve(
    repo: BaseTradesRepository,
    date_from_ms: int | None = None,
    date_to_ms: int | None = None,
    symbols: list[str] | None = None,
    direction: str | None = None,
    exit_reasons: list[str] | None = None,
) -> list[dict]:
    """Получить кривую эквити для графика с учетом фильтров."""
    initial = _get_initial_balance()
    rows = _fetch_closed_trades(
        repo,
        columns=["exit_ts_ms", "pnl_quote"],
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols,
        direction=direction,
        exit_reasons=exit_reasons,
        order_desc=False,
    )

    equity = []
    balance = initial
    for row in rows:
        balance += row["pnl_quote"]
        equity.append(
            {
                "time": datetime.fromtimestamp(
                    row["exit_ts_ms"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
                "balance": round(balance, 2),
            }
        )
    return equity


def _build_equity_curve_from_rows(rows: list[dict]) -> list[dict]:
    """Собрать equity curve из уже загруженных закрытых сделок."""
    initial = _get_initial_balance()
    equity: list[dict] = []
    balance = initial
    for row in rows:
        balance += row["pnl_quote"]
        equity.append(
            {
                "time": datetime.fromtimestamp(
                    row["exit_ts_ms"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
                "balance": round(balance, 2),
            }
        )
    return equity


def _get_request_filters() -> tuple[str, str, list[str], str, list[str], int | None, int | None]:
    """Прочитать фильтры из query string и сразу подготовить timestamps."""
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    symbols = request.args.getlist("symbols")
    direction = request.args.get("direction", "")
    exit_reasons = request.args.getlist("exit_reasons")
    date_from_ms = _parse_date(date_from) if date_from else None
    date_to_ms = _parse_date(date_to) if date_to else None
    return date_from, date_to, symbols, direction, exit_reasons, date_from_ms, date_to_ms


def _get_demo_recent_trades() -> list[dict]:
    """Небольшой демо-набор сделок для превью интерфейса."""
    now = datetime.now(timezone.utc)
    raw_trades = [
        ("BTC/USDT", "LONG", 68420.0, 69180.0, 120.0, 8.40, 7.0, "TP", True, 20),
        ("ETH/USDT", "SHORT", 3528.0, 3562.0, 90.0, -3.15, -3.5, "SL", False, 16),
        ("SOL/USDT", "LONG", 182.4, 186.7, 85.0, 5.95, 7.0, "TP", True, 12),
        ("XRP/USDT", "SHORT", 0.6421, 0.6355, 70.0, 2.87, 4.1, "manual", True, 8),
        ("BNB/USDT", "LONG", 602.3, 598.4, 95.0, -2.10, -2.2, "timeout", False, 4),
    ]
    trades: list[dict] = []
    for symbol, direction, entry_price, exit_price, notional, pnl, pnl_pct, reason, is_win, hours_ago in raw_trades:
        trades.append(
            {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": notional,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "exit_time": (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M UTC"),
                "is_win": is_win,
            }
        )
    return trades


def _get_demo_equity_curve(initial_balance: float) -> list[dict]:
    """Демо-кривая эквити для пустого состояния."""
    now = datetime.now(timezone.utc)
    balances = [initial_balance, initial_balance + 3.2, initial_balance + 1.1, initial_balance + 7.0, initial_balance + 9.8, initial_balance + 7.7, initial_balance + 11.97]
    equity: list[dict] = []
    for idx, balance in enumerate(balances):
        point_time = now - timedelta(hours=(len(balances) - idx) * 4)
        equity.append(
            {
                "time": point_time.strftime("%Y-%m-%d %H:%M"),
                "balance": round(balance, 2),
            }
        )
    return equity


# HTML шаблон с современным дизайном и фильтрами
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Trading Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; }
        .glass-panel {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
        }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .chart-container { position: relative; height: 300px; }
        .trade-row:hover { background: rgba(99, 102, 241, 0.1); }
        .filter-active {
            background: rgba(99, 102, 241, 0.3) !important;
            border-color: rgba(99, 102, 241, 0.6) !important;
        }
        .filter-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            background: rgba(99, 102, 241, 0.2);
            border-radius: 4px;
            font-size: 11px;
            color: #818cf8;
        }

        /* Кастомный сингл селект */
        .custom-select {
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            background-color: #1e293b;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236366f1' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            padding-right: 32px;
            color: #e2e8f0;
            border: 1px solid #334155;
            border-radius: 8px;
            font-size: 13px;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .custom-select:hover {
            border-color: #6366f1;
            background-color: #252f47;
        }
        .custom-select:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        /* Стили для date input */
        .custom-date {
            background-color: #1e293b;
            color: #e2e8f0;
            border: 1px solid #334155;
            border-radius: 8px;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        .custom-date:hover {
            border-color: #6366f1;
            background-color: #252f47;
        }
        .custom-date:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        .custom-date::-webkit-calendar-picker-indicator {
            filter: invert(0.6);
            cursor: pointer;
            transition: filter 0.2s;
        }
        .custom-date::-webkit-calendar-picker-indicator:hover {
            filter: invert(0.8) sepia(1) saturate(5) hue-rotate(200deg);
        }

        /* Кастомный мультиселект с чипсами */
        .multi-select-container {
            position: relative;
        }
        .multi-select-trigger {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            padding: 8px 12px;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
            min-height: 40px;
        }
        .multi-select-trigger:hover {
            border-color: #6366f1;
            background-color: #252f47;
        }
        .multi-select-trigger.active {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        .multi-select-trigger .placeholder {
            color: #64748b;
        }
        .multi-select-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            margin-top: 4px;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 100;
            display: none;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        }
        .multi-select-dropdown.open {
            display: block;
        }
        .multi-select-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            cursor: pointer;
            transition: background-color 0.15s;
            font-size: 13px;
        }
        .multi-select-option:hover {
            background-color: rgba(99, 102, 241, 0.2);
        }
        .multi-select-option.selected {
            background-color: rgba(99, 102, 241, 0.3);
            color: #fff;
        }
        .multi-select-option input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: #6366f1;
            cursor: pointer;
        }
        .selected-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            max-width: calc(100% - 24px);
            align-items: center;
            max-height: 52px;
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 4px;
            scrollbar-width: thin;
            scrollbar-color: rgba(99, 102, 241, 0.6) transparent;
            align-content: flex-start;
        }
        .selected-chips::-webkit-scrollbar {
            width: 6px;
        }
        .selected-chips::-webkit-scrollbar-track {
            background: transparent;
        }
        .selected-chips::-webkit-scrollbar-thumb {
            background-color: rgba(99, 102, 241, 0.45);
            border-radius: 999px;
        }
        .selected-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            background-color: rgba(99, 102, 241, 0.3);
            border: 1px solid rgba(99, 102, 241, 0.5);
            border-radius: 4px;
            font-size: 12px;
            color: #fff;
            max-width: 100%;
            white-space: nowrap;
        }
        .selected-chip .remove {
            cursor: pointer;
            color: #94a3b8;
            font-size: 14px;
            line-height: 1;
        }
        .selected-chip .remove:hover {
            color: #fff;
        }
        .selected-chip-summary {
            background-color: rgba(148, 163, 184, 0.16);
            border-color: rgba(148, 163, 184, 0.28);
            color: #cbd5e1;
        }
        .dropdown-arrow {
            color: #6366f1;
            font-size: 12px;
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        .multi-select-trigger.active .dropdown-arrow {
            transform: rotate(180deg);
        }

        /* Лейблы форм */
        .form-label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #94a3b8;
            font-weight: 500;
            margin-bottom: 6px;
        }
        .form-label i {
            color: #6366f1;
            font-size: 10px;
        }
    </style>
</head>
<body class="text-slate-200 min-h-screen">
    <!-- Header -->
    <header class="glass-panel sticky top-0 z-50 border-b border-slate-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <i class="fas fa-chart-line text-white text-lg"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-white">Paper Trading Dashboard</h1>
                        <p class="text-xs text-slate-400">Live Trading Analytics</p>
                    </div>
                </div>
                <div class="flex items-center gap-4 text-sm">
                    <div class="px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700">
                        <i class="fas fa-exchange-alt text-indigo-400 mr-2"></i>
                        <span class="text-slate-300">{{ metrics.exchange }}</span>
                    </div>
                    <div class="px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700">
                        <i class="fas fa-clock text-indigo-400 mr-2"></i>
                        <span class="text-slate-300">{{ metrics.timeframe }} / {{ metrics.htf_timeframe }}</span>
                    </div>
                    <div class="px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700">
                        <i class="fas fa-expand-arrows-alt text-indigo-400 mr-2"></i>
                        <span class="text-slate-300">{{ metrics.leverage }}x</span>
                    </div>
                    <button onclick="location.reload()" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors">
                        <i class="fas fa-sync-alt mr-1"></i> Обновить
                    </button>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <!-- Filters Panel -->
        <div class="glass-panel rounded-xl p-5 mb-6">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold text-white">
                    <i class="fas fa-filter mr-2 text-indigo-400"></i>Фильтры
                </h3>
                <div class="flex gap-2">
                    {% if has_active_filters %}
                    <a href="/" class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-colors">
                        <i class="fas fa-times mr-1"></i> Сбросить
                    </a>
                    {% endif %}
                    <button onclick="applyFilters()" class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-medium transition-colors">
                        <i class="fas fa-check mr-1"></i> Применить
                    </button>
                </div>
            </div>

            <form id="filterForm" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                <!-- Date From -->
                <div>
                    <label class="form-label">
                        <i class="fas fa-calendar-alt"></i> Дата от
                    </label>
                    <input type="date" name="date_from" value="{{ date_from }}"
                           class="w-full px-3 py-2.5 custom-date">
                </div>

                <!-- Date To -->
                <div>
                    <label class="form-label">
                        <i class="fas fa-calendar-alt"></i> Дата до
                    </label>
                    <input type="date" name="date_to" value="{{ date_to }}"
                           class="w-full px-3 py-2.5 custom-date">
                </div>

                <!-- Symbols Multi-Select -->
                <div>
                    <label class="form-label">
                        <i class="fas fa-coins"></i> Монеты
                    </label>
                    <div class="multi-select-container" id="symbolsContainer">
                        <div class="multi-select-trigger" onclick="toggleDropdown('symbols')">
                            <div class="selected-chips" id="symbolsChips">
                                {% if selected_symbols %}
                                    {% for sym in selected_symbols %}
                                    <span class="selected-chip" data-value="{{ sym }}">{{ sym }}<span class="remove" data-value="{{ sym }}" onclick="removeChip(event, 'symbols', this.dataset.value)">×</span></span>
                                    {% endfor %}
                                {% else %}
                                    <span class="placeholder">Все монеты</span>
                                {% endif %}
                            </div>
                            <i class="fas fa-chevron-down dropdown-arrow"></i>
                        </div>
                        <div class="multi-select-dropdown" id="symbolsDropdown">
                            {% for sym in all_symbols %}
                            <div class="multi-select-option {% if sym in selected_symbols %}selected{% endif %}" data-value="{{ sym }}" onclick="toggleOption(event, 'symbols', this.dataset.value, this)">
                                <input type="checkbox" value="{{ sym }}" {% if sym in selected_symbols %}checked{% endif %} onclick="event.stopPropagation()" onchange="handleCheckboxChange('symbols', this)">
                                <span>{{ sym }}</span>
                            </div>
                            {% endfor %}
                        </div>
                        <input type="hidden" name="symbols" id="symbolsInput" value="{{ selected_symbols|join(',') }}">
                    </div>
                </div>

                <!-- Direction -->
                <div>
                    <label class="form-label">
                        <i class="fas fa-exchange-alt"></i> Направление
                    </label>
                    <select name="direction" class="w-full px-3 py-2.5 custom-select">
                        <option value="" {% if not direction %}selected{% endif %}>Все направления</option>
                        <option value="long" {% if direction == 'long' %}selected{% endif %}>🟢 LONG</option>
                        <option value="short" {% if direction == 'short' %}selected{% endif %}>🔴 SHORT</option>
                    </select>
                </div>

                <!-- Exit Reasons Multi-Select -->
                <div>
                    <label class="form-label">
                        <i class="fas fa-door-open"></i> Причина выхода
                    </label>
                    <div class="multi-select-container" id="exitReasonsContainer">
                        <div class="multi-select-trigger" onclick="toggleDropdown('exitReasons')">
                            <div class="selected-chips" id="exitReasonsChips">
                                {% if selected_exit_reasons %}
                                    {% for reason in selected_exit_reasons %}
                                    <span class="selected-chip" data-value="{{ reason }}">{{ reason }}<span class="remove" data-value="{{ reason }}" onclick="removeChip(event, 'exitReasons', this.dataset.value)">×</span></span>
                                    {% endfor %}
                                {% else %}
                                    <span class="placeholder">Все причины</span>
                                {% endif %}
                            </div>
                            <i class="fas fa-chevron-down dropdown-arrow"></i>
                        </div>
                        <div class="multi-select-dropdown" id="exitReasonsDropdown">
                            {% for reason in all_exit_reasons %}
                            <div class="multi-select-option {% if reason in selected_exit_reasons %}selected{% endif %}" data-value="{{ reason }}" onclick="toggleOption(event, 'exitReasons', this.dataset.value, this)">
                                <input type="checkbox" value="{{ reason }}" {% if reason in selected_exit_reasons %}checked{% endif %} onclick="event.stopPropagation()" onchange="handleCheckboxChange('exitReasons', this)">
                                <span>{{ reason }}</span>
                            </div>
                            {% endfor %}
                        </div>
                        <input type="hidden" name="exit_reasons" id="exitReasonsInput" value="{{ selected_exit_reasons|join(',') }}">
                    </div>
                </div>
            </form>

            <!-- Active Filters Display -->
            {% if has_active_filters %}
            <div class="mt-4 pt-4 border-t border-slate-700/50">
                <div class="flex items-center gap-2 flex-wrap">
                    <span class="text-xs text-slate-400">Активные фильтры:</span>
                    {% if date_from %}<span class="filter-badge">От: {{ date_from }}</span>{% endif %}
                    {% if date_to %}<span class="filter-badge">До: {{ date_to }}</span>{% endif %}
                    {% if selected_symbols %}<span class="filter-badge">Монеты: {{ selected_symbols|length }}</span>{% endif %}
                    {% if direction %}<span class="filter-badge">{{ direction|upper }}</span>{% endif %}
                    {% if selected_exit_reasons %}<span class="filter-badge">Выход: {{ selected_exit_reasons|join(', ') }}</span>{% endif %}
                </div>
            </div>
            {% endif %}
        </div>

        <!-- Filtered Stats Banner -->
        {% if has_active_filters %}
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div class="metric-card rounded-xl p-4 border-l-4 border-indigo-500">
                <div class="text-xs text-slate-400 mb-1">Отфильтрованный P&L</div>
                <div class="text-xl font-bold {% if metrics.filtered_realized_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "+" if metrics.filtered_realized_pnl >= 0 else "" }}${{ "%.2f"|format(metrics.filtered_realized_pnl) }}
                </div>
            </div>
            <div class="metric-card rounded-xl p-4 border-l-4 border-purple-500">
                <div class="text-xs text-slate-400 mb-1">Отфильтрованный Win Rate</div>
                <div class="text-xl font-bold {% if metrics.filtered_win_rate >= 50 %}positive{% else %}negative{% endif %}">
                    {{ "%.1f"|format(metrics.filtered_win_rate) }}%
                </div>
            </div>
            <div class="metric-card rounded-xl p-4 border-l-4 border-amber-500">
                <div class="text-xs text-slate-400 mb-1">Отфильтровано сделок</div>
                <div class="text-xl font-bold text-white">{{ metrics.filtered_trades_count }}</div>
            </div>
        </div>
        {% endif %}

        <!-- Balance Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div class="metric-card rounded-xl p-5">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-slate-400 text-sm font-medium">Баланс кошелька</span>
                    <i class="fas fa-wallet text-indigo-400"></i>
                </div>
                <div class="text-2xl font-bold text-white">${{ "%.2f"|format(metrics.wallet_balance) }}</div>
                <div class="text-xs mt-1 {% if metrics.realized_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    P&L: {{ "+%.2f"|format(metrics.realized_pnl) if metrics.realized_pnl >= 0 else "%.2f"|format(metrics.realized_pnl) }}
                </div>
            </div>

            <div class="metric-card rounded-xl p-5">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-slate-400 text-sm font-medium">Доступно</span>
                    <i class="fas fa-unlock text-emerald-400"></i>
                </div>
                <div class="text-2xl font-bold text-white">${{ "%.2f"|format(metrics.available_balance) }}</div>
                <div class="text-xs text-slate-400 mt-1">Маржа занята: ${{ "%.2f"|format(metrics.open_margin) }}</div>
            </div>

            <div class="metric-card rounded-xl p-5">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-slate-400 text-sm font-medium">Открытые позиции</span>
                    <i class="fas fa-chart-pie text-amber-400"></i>
                </div>
                <div class="text-2xl font-bold text-white">{{ metrics.open_positions_count }} / {{ metrics.total_trades }}</div>
                <div class="text-xs text-slate-400 mt-1">
                    LONG: {{ metrics.longs_count }} | SHORT: {{ metrics.shorts_count }}
                </div>
            </div>

            <div class="metric-card rounded-xl p-5">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-slate-400 text-sm font-medium">Win Rate</span>
                    <i class="fas fa-percentage text-purple-400"></i>
                </div>
                <div class="text-2xl font-bold {% if metrics.win_rate >= 50 %}positive{% else %}negative{% endif %}">
                    {{ "%.1f"|format(metrics.win_rate) }}%
                </div>
                <div class="text-xs text-slate-400 mt-1">Profit Factor: {{ "%.2f"|format(metrics.profit_factor) }}</div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Column: Stats & Chart -->
            <div class="lg:col-span-2 space-y-6">
                {% if demo_mode %}
                <div class="glass-panel rounded-xl p-4 border border-amber-400/30 bg-amber-500/10">
                    <div class="flex items-center gap-2 text-amber-300 text-sm font-medium">
                        <i class="fas fa-flask"></i>
                        <span>{{ demo_message }}</span>
                    </div>
                </div>
                {% endif %}

                <!-- Equity Chart -->
                <div class="glass-panel rounded-xl p-5">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold text-white">Кривая эквити {% if has_active_filters %}(отфильтрованная){% endif %}{% if demo_mode %} <span class="text-amber-300 text-sm font-medium">(демо)</span>{% endif %}</h3>
                        <span class="text-xs text-slate-400">Начальный баланс: ${{ "%.2f"|format(metrics.initial_balance) }}</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>

                <!-- Statistics Grid -->
                <div class="glass-panel rounded-xl p-5">
                    <h3 class="text-lg font-semibold text-white mb-4">Статистика торговли</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Avg Win</div>
                            <div class="text-lg font-semibold positive">${{ "%.2f"|format(metrics.avg_win) }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Avg Loss</div>
                            <div class="text-lg font-semibold negative">${{ "%.2f"|format(metrics.avg_loss) }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Best Trade</div>
                            <div class="text-lg font-semibold positive">${{ "%.2f"|format(metrics.best_trade) }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Worst Trade</div>
                            <div class="text-lg font-semibold negative">${{ "%.2f"|format(metrics.worst_trade) }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Max Drawdown</div>
                            <div class="text-lg font-semibold text-orange-400">{{ "%.2f"|format(metrics.max_drawdown_pct) }}%</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Всего сделок</div>
                            <div class="text-lg font-semibold text-white">{{ metrics.total_trades }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Закрытые</div>
                            <div class="text-lg font-semibold text-white">{{ metrics.closed_trades_count }}</div>
                        </div>
                        <div class="bg-slate-800/50 rounded-lg p-3">
                            <div class="text-xs text-slate-400">Начальный баланс</div>
                            <div class="text-lg font-semibold text-white">${{ "%.2f"|format(metrics.initial_balance) }}</div>
                        </div>
                    </div>
                </div>

                <!-- Recent Trades -->
                <div class="glass-panel rounded-xl p-5">
                    <h3 class="text-lg font-semibold text-white mb-4">{% if has_active_filters %}Отфильтрованные сделки{% else %}Последние сделки{% endif %}{% if demo_mode %} <span class="text-amber-300 text-sm font-medium">(демо)</span>{% endif %}</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="text-left text-slate-400 border-b border-slate-700">
                                    <th class="pb-2">Символ</th>
                                    <th class="pb-2">Направление</th>
                                    <th class="pb-2">Вход</th>
                                    <th class="pb-2">Выход</th>
                                    <th class="pb-2">P&L</th>
                                    <th class="pb-2">Причина</th>
                                    <th class="pb-2">Время</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for trade in recent_trades %}
                                <tr class="trade-row border-b border-slate-800/50 transition-colors">
                                    <td class="py-3 font-medium">{{ trade.symbol }}</td>
                                    <td class="py-3">
                                        <span class="px-2 py-1 rounded text-xs {% if trade.direction == 'LONG' %}bg-emerald-500/20 text-emerald-400{% else %}bg-red-500/20 text-red-400{% endif %}">
                                            {{ trade.direction }}
                                        </span>
                                    </td>
                                    <td class="py-3">${{ "%.6f"|format(trade.entry_price) if trade.entry_price < 1 else "%.2f"|format(trade.entry_price) }}</td>
                                    <td class="py-3">${{ "%.6f"|format(trade.exit_price) if trade.exit_price < 1 else "%.2f"|format(trade.exit_price) }}</td>
                                    <td class="py-3 font-semibold {% if trade.is_win %}positive{% else %}negative{% endif %}">
                                        {{ "+" if trade.is_win else "" }}${{ "%.2f"|format(trade.pnl) }}
                                    </td>
                                    <td class="py-3 text-slate-400">{{ trade.exit_reason }}</td>
                                    <td class="py-3 text-slate-400 text-xs">{{ trade.exit_time }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Right Column: Open Positions -->
            <div class="space-y-6">
                <div class="glass-panel rounded-xl p-5">
                    <h3 class="text-lg font-semibold text-white mb-4">
                        <i class="fas fa-briefcase mr-2 text-indigo-400"></i>
                        Открытые позиции
                        <span class="ml-2 text-sm font-normal text-slate-400">({{ open_positions|length }})</span>
                    </h3>
                    {% if open_positions %}
                        <div class="space-y-3">
                            {% for pos in open_positions %}
                            <div class="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                                <div class="flex items-center justify-between mb-2">
                                    <span class="font-semibold text-white">{{ pos.symbol }}</span>
                                    <span class="px-2 py-1 rounded text-xs {% if pos.direction == 'LONG' %}bg-emerald-500/20 text-emerald-400{% else %}bg-red-500/20 text-red-400{% endif %}">
                                        {{ pos.direction }}
                                    </span>
                                </div>
                                <div class="grid grid-cols-2 gap-2 text-xs text-slate-400">
                                    <div>Вход: <span class="text-slate-200">${{ "%.6f"|format(pos.entry_price) if pos.entry_price < 1 else "%.2f"|format(pos.entry_price) }}</span></div>
                                    <div>Размер: <span class="text-slate-200">${{ "%.2f"|format(pos.entry_notional) }}</span></div>
                                    <div>SL: <span class="text-red-400">{{ "%.2f"|format(pos.stop_pct) }}%</span></div>
                                    <div>TP: <span class="text-emerald-400">{{ "%.2f"|format(pos.take_pct) }}%</span></div>
                                </div>
                                <div class="mt-2 pt-2 border-t border-slate-700/50 flex items-center justify-between text-xs">
                                    <span class="text-slate-500">P(LONG): {{ "%.3f"|format(pos.p_long) if pos.p_long is not none else "n/a" }}</span>
                                    <span class="text-slate-500">P(SHORT): {{ "%.3f"|format(pos.p_short) if pos.p_short is not none else "n/a" }}</span>
                                </div>
                                <div class="mt-1 text-xs text-slate-500">{{ pos.entry_time }}</div>
                            </div>
                            {% endfor %}
                        </div>
                    {% else %}
                        <div class="text-center py-8 text-slate-500">
                            <i class="fas fa-inbox text-3xl mb-2"></i>
                            <p>Нет открытых позиций</p>
                        </div>
                    {% endif %}
                </div>

                <!-- Quick Info -->
                <div class="glass-panel rounded-xl p-5">
                    <h3 class="text-lg font-semibold text-white mb-4">Информация</h3>
                    <div class="space-y-3 text-sm">
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400">Биржа</span>
                            <span class="text-white font-medium">{{ metrics.exchange }}</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400">Таймфрейм</span>
                            <span class="text-white font-medium">{{ metrics.timeframe }}</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400">HTF Таймфрейм</span>
                            <span class="text-white font-medium">{{ metrics.htf_timeframe }}</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400">Плечо</span>
                            <span class="text-white font-medium">{{ metrics.leverage }}x</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400">Начальный баланс</span>
                            <span class="text-white font-medium">${{ "%.2f"|format(metrics.initial_balance) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-slate-500 text-sm">
        <p>Paper Trading System | Обновлено: {{ current_time }}</p>
    </footer>

    <script>
        // ===== Multi-Select Dropdown Functions =====
        const openDropdowns = new Set();

        function toggleDropdown(type) {
            const dropdown = document.getElementById(type + 'Dropdown');
            const trigger = document.querySelector('#' + type + 'Container .multi-select-trigger');

            if (openDropdowns.has(type)) {
                dropdown.classList.remove('open');
                trigger.classList.remove('active');
                openDropdowns.delete(type);
            } else {
                // Close all other dropdowns
                openDropdowns.forEach(t => {
                    document.getElementById(t + 'Dropdown').classList.remove('open');
                    document.querySelector('#' + t + 'Container .multi-select-trigger').classList.remove('active');
                });
                openDropdowns.clear();

                dropdown.classList.add('open');
                trigger.classList.add('active');
                openDropdowns.add(type);
            }
        }

        function updateChip(type, value, checked) {
            const chipsContainer = document.getElementById(type + 'Chips');
            const input = document.getElementById(type + 'Input');

            let selectedValues = input.value ? input.value.split(',') : [];

            if (checked) {
                if (!selectedValues.includes(value)) {
                    selectedValues.push(value);
                }
            } else {
                selectedValues = selectedValues.filter(v => v !== value);
            }

            input.value = selectedValues.join(',');
            renderChips(type, selectedValues);
        }

        function renderChips(type, values) {
            const chipsContainer = document.getElementById(type + 'Chips');
            const visibleLimit = type === 'symbols' ? 3 : 2;

            chipsContainer.title = values.join(', ');

            if (values.length === 0) {
                chipsContainer.innerHTML = '<span class="placeholder">' + (type === 'symbols' ? 'Все монеты' : 'Все причины') + '</span>';
            } else {
                chipsContainer.innerHTML = '';
                const visibleValues = values.slice(0, visibleLimit);

                visibleValues.forEach(v => {
                    const chip = document.createElement('span');
                    chip.className = 'selected-chip';
                    chip.dataset.value = v;
                    chip.appendChild(document.createTextNode(v));

                    const remove = document.createElement('span');
                    remove.className = 'remove';
                    remove.dataset.value = v;
                    remove.textContent = '×';
                    remove.addEventListener('click', event => removeChip(event, type, v));

                    chip.appendChild(remove);
                    chipsContainer.appendChild(chip);
                });

                const hiddenCount = values.length - visibleValues.length;
                if (hiddenCount > 0) {
                    const summaryChip = document.createElement('span');
                    summaryChip.className = 'selected-chip selected-chip-summary';
                    summaryChip.textContent = '+' + hiddenCount + ' еще';
                    chipsContainer.appendChild(summaryChip);
                }
            }
        }

        function syncOptionState(element, checked) {
            if (checked) {
                element.classList.add('selected');
            } else {
                element.classList.remove('selected');
            }
        }

        function handleCheckboxChange(type, checkbox) {
            const option = checkbox.closest('.multi-select-option');
            syncOptionState(option, checkbox.checked);
            updateChip(type, checkbox.value, checkbox.checked);
        }

        function toggleOption(event, type, value, element) {
            if (event.target.closest('input[type="checkbox"]')) {
                return;
            }

            const checkbox = element.querySelector('input[type="checkbox"]');
            checkbox.checked = !checkbox.checked;
            syncOptionState(element, checkbox.checked);
            updateChip(type, value, checkbox.checked);
        }

        function removeChip(event, type, value) {
            event.stopPropagation();

            const input = document.getElementById(type + 'Input');
            let selectedValues = input.value ? input.value.split(',') : [];
            selectedValues = selectedValues.filter(v => v !== value);
            input.value = selectedValues.join(',');

            renderChips(type, selectedValues);

            // Uncheck in dropdown
            const option = document.querySelector('#' + type + 'Dropdown .multi-select-option[data-value="' + value + '"]');
            if (option) {
                const checkbox = option.querySelector('input[type="checkbox"]');
                if (checkbox) {
                    checkbox.checked = false;
                    option.classList.remove('selected');
                }
            }
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.multi-select-container')) {
                openDropdowns.forEach(type => {
                    document.getElementById(type + 'Dropdown').classList.remove('open');
                    document.querySelector('#' + type + 'Container .multi-select-trigger').classList.remove('active');
                });
                openDropdowns.clear();
            }
        });

        document.addEventListener('DOMContentLoaded', function() {
            const symbolsInput = document.getElementById('symbolsInput');
            const exitReasonsInput = document.getElementById('exitReasonsInput');

            renderChips('symbols', symbolsInput && symbolsInput.value ? symbolsInput.value.split(',').filter(Boolean) : []);
            renderChips('exitReasons', exitReasonsInput && exitReasonsInput.value ? exitReasonsInput.value.split(',').filter(Boolean) : []);
        });

        // ===== Apply Filters =====
        function applyFilters() {
            const params = new URLSearchParams();

            // Date filters
            const dateFrom = document.querySelector('input[name="date_from"]').value;
            const dateTo = document.querySelector('input[name="date_to"]').value;
            if (dateFrom) params.append('date_from', dateFrom);
            if (dateTo) params.append('date_to', dateTo);

            // Symbols
            const symbolsInput = document.getElementById('symbolsInput').value;
            if (symbolsInput) {
                symbolsInput.split(',').forEach(s => {
                    if (s.trim()) params.append('symbols', s.trim());
                });
            }

            // Direction
            const direction = document.querySelector('select[name="direction"]').value;
            if (direction) params.append('direction', direction);

            // Exit reasons
            const exitReasonsInput = document.getElementById('exitReasonsInput').value;
            if (exitReasonsInput) {
                exitReasonsInput.split(',').forEach(r => {
                    if (r.trim()) params.append('exit_reasons', r.trim());
                });
            }

            window.location.href = '/?' + params.toString();
        }

        // ===== Equity Chart =====
        const equityData = {{ equity_curve|tojson }};
        const ctx = document.getElementById('equityChart').getContext('2d');

        const labels = equityData.map(d => d.time);
        const data = equityData.map(d => d.balance);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Баланс',
                    data: data,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#94a3b8',
                        bodyColor: '#e2e8f0',
                        borderColor: 'rgba(99, 102, 241, 0.3)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                return 'Balance: $' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#64748b',
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""


def index():
    """Главная страница дашборда."""
    repo = _get_repo()
    (
        date_from,
        date_to,
        symbols,
        direction,
        exit_reasons,
        date_from_ms,
        date_to_ms,
    ) = _get_request_filters()

    # Определяем есть ли активные фильтры
    has_active_filters = bool(date_from or date_to or symbols or direction or exit_reasons)

    open_trades = repo.list_open_trades(EXEC_TYPE)
    filtered_closed_rows = _fetch_closed_trades(
        repo,
        columns=[
            "symbol",
            "direction",
            "entry_price",
            "exit_price",
            "entry_notional",
            "pnl_quote",
            "pnl_pct",
            "exit_reason",
            "exit_ts_ms",
        ],
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols if symbols else None,
        direction=direction if direction else None,
        exit_reasons=exit_reasons if exit_reasons else None,
        order_desc=False,
    )
    if has_active_filters:
        all_closed_rows = _fetch_closed_trades(
            repo,
            columns=["direction", "pnl_quote", "pnl_pct", "exit_reason"],
            order_desc=False,
        )
    else:
        all_closed_rows = filtered_closed_rows

    # Получаем данные с фильтрами
    metrics = calculate_metrics(
        repo,
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols if symbols else None,
        direction=direction if direction else None,
        exit_reasons=exit_reasons if exit_reasons else None,
        open_trades=open_trades,
        all_closed_trades=all_closed_rows,
        filtered_closed_trades=filtered_closed_rows,
    )
    open_positions = get_open_positions(repo, open_trades)
    recent_trades = _format_recent_trades(filtered_closed_rows, limit=100)
    equity_curve = _build_equity_curve_from_rows(filtered_closed_rows)
    demo_mode = False
    demo_message = ""
    if not recent_trades and not equity_curve:
        recent_trades = _get_demo_recent_trades()
        equity_curve = _get_demo_equity_curve(metrics.initial_balance)
        demo_mode = True
        demo_message = (
            "По текущим фильтрам данных нет, показан демо-превью графика и таблицы сделок"
            if has_active_filters
            else "Показаны демо-данные для превью графика и таблицы сделок"
        )

    # Данные для фильтров
    all_symbols = _get_available_symbols()
    all_exit_reasons = _get_exit_reasons(repo)

    return render_template_string(
        HTML_TEMPLATE,
        metrics=metrics,
        open_positions=open_positions,
        recent_trades=recent_trades,
        equity_curve=equity_curve,
        all_symbols=all_symbols,
        all_exit_reasons=all_exit_reasons,
        date_from=date_from,
        date_to=date_to,
        selected_symbols=symbols,
        direction=direction,
        selected_exit_reasons=exit_reasons,
        has_active_filters=has_active_filters,
        demo_mode=demo_mode,
        demo_message=demo_message,
        current_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


app.add_url_rule("/", "index", index)


@app.route("/api/metrics")
def api_metrics():
    """JSON API для метрик."""
    repo = _get_repo()
    _, _, symbols, direction, exit_reasons, date_from_ms, date_to_ms = _get_request_filters()

    metrics = calculate_metrics(
        repo,
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols if symbols else None,
        direction=direction if direction else None,
        exit_reasons=exit_reasons if exit_reasons else None,
    )

    return jsonify(
        {
            "exchange": metrics.exchange,
            "timeframe": metrics.timeframe,
            "htf_timeframe": metrics.htf_timeframe,
            "leverage": metrics.leverage,
            "initial_balance": metrics.initial_balance,
            "wallet_balance": metrics.wallet_balance,
            "realized_pnl": metrics.realized_pnl,
            "available_balance": metrics.available_balance,
            "open_positions_count": metrics.open_positions_count,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "total_trades": metrics.total_trades,
            "filtered_pnl": metrics.filtered_realized_pnl,
            "filtered_win_rate": metrics.filtered_win_rate,
            "filtered_count": metrics.filtered_trades_count,
        }
    )


@app.route("/api/positions")
def api_positions():
    """JSON API для открытых позиций."""
    repo = _get_repo()
    return jsonify(get_open_positions(repo))


@app.route("/api/trades")
def api_trades():
    """JSON API для сделок с фильтрами."""
    repo = _get_repo()
    _, _, symbols, direction, exit_reasons, date_from_ms, date_to_ms = _get_request_filters()
    limit = min(int(request.args.get("limit", 50)), 200)

    trades = get_filtered_trades(
        repo,
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols if symbols else None,
        direction=direction if direction else None,
        exit_reasons=exit_reasons if exit_reasons else None,
        limit=limit,
    )
    return jsonify(trades)


@app.route("/api/equity")
def api_equity():
    """JSON API для кривой эквити с фильтрами."""
    repo = _get_repo()
    _, _, symbols, direction, exit_reasons, date_from_ms, date_to_ms = _get_request_filters()

    equity = get_equity_curve(
        repo,
        date_from_ms=date_from_ms,
        date_to_ms=date_to_ms,
        symbols=symbols if symbols else None,
        direction=direction if direction else None,
        exit_reasons=exit_reasons if exit_reasons else None,
    )
    return jsonify(equity)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    db_type = str(getattr(cfg, "EXECUTION_DB_TYPE", "sqlite")).lower()
    logger.info("Инициализация репозитория сделок (%s)...", db_type)
    _get_repo()
    logger.info("✅ Репозиторий сделок инициализирован (%s)", db_type)
    logger.info("Запуск веб-дашборда на http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
