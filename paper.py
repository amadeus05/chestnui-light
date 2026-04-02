"""
Пейпер-трейдинг: синхронизация по BTC (основной ТФ + 1m), сделки в SQLite.

- BTC 1h: определяем момент «закрытся час на бирже» и запускаем анализ всех SYMBOLS.
- BTC 1m: тот же «биржевой» тайминг между часовыми прогонами (последняя закрытая минута).
- Выходы TP/SL внутри часа: по 1m свечам **торгуемого символа** (иначе нельзя
  восстановить порядок касаний для альтов). BTC 1m здесь не используется.

Таблица execution_trades: execution_type IN ('paper','live') — для live потом тот же журнал.

Запуск:
  python paper.py           # по умолчанию: цикл до Ctrl+C (то же, что paper.py daemon)
  python paper.py tick      # один проход (cron / ручная проверка)
  python paper.py status    # открытые + баланс
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as cfg
import etl
from bt import build_entry_score, compute_trade_outcome, resolve_directional_signal
from signal_filter import build_candidate_event_mask, resolve_event_filter_config
from src.features import MasterFeatureBuilder
from src.persistence.repositories.base_trades_repository import BaseTradesRepository
from src.persistence.repositories.trades_repository_factory import create_trades_repository_from_config

logger = logging.getLogger("paper")

STATE_LAST_HOUR = "paper_last_processed_main_bar_open_ms_btc"
EXEC_TYPE = "paper"


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _series_open_ms(ts: pd.Series) -> np.ndarray:
    return (ts.astype("int64") // 1_000_000).to_numpy(dtype=np.int64)


def _mask_fully_closed(df: pd.DataFrame, tf_ms: int, now_ms: int) -> pd.Series:
    o = _series_open_ms(df["timestamp"])
    return pd.Series(o + tf_ms <= now_ms, index=df.index)


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    rows = [
        (k.open_time, k.open, k.high, k.low, k.close, k.volume)
        for k in sorted(klines, key=lambda x: x.open_time)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _fetch_ohlcv_bars(exchange, symbol: str, timeframe: str, min_bars: int) -> pd.DataFrame:
    tf_ms = exchange.get_timeframe_ms(timeframe)
    end_ts = _utc_now_ms()
    start_ts = end_ts - int(min_bars * tf_ms)
    klines = exchange.fetch_klines(symbol, timeframe, start_ts, end_ts)
    return _klines_to_dataframe(klines)


def _last_closed_bar_open_ms(df: pd.DataFrame, tf_ms: int, now_ms: int) -> int | None:
    if df is None or df.empty:
        return None
    m = _mask_fully_closed(df, tf_ms, now_ms)
    sub = df.loc[m]
    if sub.empty:
        return None
    o = _series_open_ms(sub["timestamp"])
    return int(o[-1])


def _normalize_features_for_model(
    latest_row: pd.DataFrame,
    feature_names: list,
    symbol_categories: list | None,
) -> pd.DataFrame:
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def _apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict) -> pd.DataFrame:
    if not clip_bounds:
        return frame
    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def _get_barrier_pcts(feature_row: pd.DataFrame) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None
    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None
    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct


def _is_candidate_event(feature_row: pd.DataFrame, event_filter_config: dict) -> bool:
    if feature_row is None or feature_row.empty:
        return False
    mask = build_candidate_event_mask(feature_row, event_filter_config)
    return bool(mask.iloc[0]) if not mask.empty else False


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _clock_symbol() -> str:
    return str(getattr(cfg, "PAPER_CLOCK_SYMBOL", "BTC/USDT"))


def _db_path() -> str:
    p = getattr(cfg, "EXECUTION_DB_PATH", None)
    if p:
        return str(p)
    return str(cfg.DB_PATH)


def _fetch_closed_1m_range(
    exchange,
    symbol: str,
    start_open_ms: int,
    end_open_ms_exclusive: int,
    tf_ms_1m: int,
) -> pd.DataFrame:
    """Свечи 1m с open_time в [start, end)."""
    if start_open_ms >= end_open_ms_exclusive:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    pad_ms = tf_ms_1m * 5
    klines = exchange.fetch_klines(symbol, "1m", start_open_ms - pad_ms, end_open_ms_exclusive + pad_ms)
    df = _klines_to_dataframe(klines)
    if df.empty:
        return df
    o = _series_open_ms(df["timestamp"])
    keep = (o >= start_open_ms) & (o < end_open_ms_exclusive)
    return df.loc[keep].reset_index(drop=True)


def _process_minute_exits_for_trade(
    exchange,
    trade,
    *,
    last_closed_1m_open_ms: int,
    now_ms: int,
    tf_1m_ms: int,
) -> bool:
    """Проверка 1m баров символа позиции. Возвращает True, если сделка закрыта."""
    repo: BaseTradesRepository = trade["repo"]
    tid = trade["id"]
    start_from = trade["last_1m_scan_open_ms"]
    if start_from is None:
        start_from = int(trade["entry_ts_ms"])

    end_open_exclusive = last_closed_1m_open_ms + tf_1m_ms
    if end_open_exclusive <= start_from:
        return False

    df = _fetch_closed_1m_range(exchange, trade["symbol"], start_from, end_open_exclusive, tf_1m_ms)
    now_ms = _utc_now_ms()
    mclosed = _mask_fully_closed(df, tf_1m_ms, now_ms)
    df = df.loc[mclosed].reset_index(drop=True)
    if df.empty:
        return False

    entry_price = float(trade["entry_price"])
    direction = int(trade["direction"])
    stop_pct = float(trade["stop_pct"])
    take_pct = float(trade["take_pct"])

    next_scan_from = start_from
    for _, row in df.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        exit_price, reason = etl.resolve_trade_exit(
            direction, entry_price, o, h, l, stop_pct, take_pct
        )
        open_ms = int(_series_open_ms(pd.Series([row["timestamp"]]))[0])
        bar_close_ms = open_ms + tf_1m_ms
        next_scan_from = bar_close_ms
        if exit_price is not None and reason:
            pos = {"dir": direction, "entry": entry_price, "size": float(trade["entry_notional"])}
            pnl_clean, trade_profit, commission = compute_trade_outcome(pos, float(exit_price))
            repo.close_trade(
                tid,
                exit_ts_ms=bar_close_ms,
                exit_price=float(exit_price),
                exit_reason=str(reason),
                pnl_pct=float(pnl_clean * 100.0),
                pnl_quote=float(trade_profit),
                fees_quote=float(commission),
            )
            logger.info(
                "CLOSE %s id=%s %s @ %.8g  %s  pnl_quote=%.2f",
                trade["symbol"],
                tid,
                "LONG" if direction == 1 else "SHORT",
                exit_price,
                reason,
                trade_profit,
            )
            return True

    repo.update_last_1m_scan(tid, next_scan_from)
    return False


def _symbol_has_open(repo: BaseTradesRepository, symbol: str) -> bool:
    for t in repo.list_open_trades(EXEC_TYPE):
        if t.symbol == symbol:
            return True
    return False


def _cooldown_blocks(
    repo: BaseTradesRepository,
    symbol: str,
    *,
    main_tf_ms: int,
    now_ms: int,
    btc_signal_open_ms: int,
) -> bool:
    bars = int(getattr(cfg, "BACKTEST_SL_COOLDOWN_BARS", 0))
    if bars <= 0:
        return False
    last_sl = repo.last_sl_exit_ms(EXEC_TYPE, symbol)
    if last_sl is None:
        return False
    if last_sl + bars * main_tf_ms > btc_signal_open_ms:
        return True
    return False


def _max_sl_day_blocks(repo: ExecutionTradesRepository, symbol: str, now_ms: int) -> bool:
    cap = int(getattr(cfg, "BACKTEST_MAX_SL_PER_DAY", 0))
    if cap <= 0:
        return False
    day = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    start = int(pd.Timestamp(f"{day}T00:00:00Z").timestamp() * 1000)
    if repo.count_sl_today_utc(EXEC_TYPE, symbol, start, now_ms) >= cap:
        return True
    return False


def run_hourly_entries(
    *,
    exchange,
    model,
    feature_names: list,
    symbol_categories: list | None,
    event_filter_config: dict,
    clip_bounds: dict,
    model_name: str,
    repo: BaseTradesRepository,
    main_tf: str,
    main_tf_ms: int,
    btc_last_closed_hour_open_ms: int,
    now_ms: int,
    main_bars: int,
    htf_bars: int,
    min_main_rows: int,
    min_htf_rows: int,
) -> None:
    scan_symbols = [str(exchange.normalize_symbol(s)) for s in dict.fromkeys(cfg.SYMBOLS)]
    base_map: dict[str, pd.DataFrame] = {}
    htf_map: dict[str, pd.DataFrame] = {}
    htf = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))

    for symbol in scan_symbols:
        try:
            main_df = _fetch_ohlcv_bars(exchange, symbol, main_tf, main_bars)
            hdf = _fetch_ohlcv_bars(exchange, symbol, htf, htf_bars)
        except Exception:
            logger.exception("Загрузка %s", symbol)
            continue
        if len(main_df) < min_main_rows or len(hdf) < min_htf_rows:
            continue
        base_map[symbol] = main_df
        htf_map[symbol] = hdf

    if not base_map:
        logger.warning("Нет символов с данными для часового тика.")
        return

    pipeline_result = MasterFeatureBuilder().build(base_map, htf_map)
    use_symbol = "symbol" in feature_names

    leverage = float(getattr(cfg, "LEVERAGE", 1))
    initial = float(getattr(cfg, "PAPER_INITIAL_BALANCE", getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)))
    balance = initial + repo.sum_closed_pnl_quote(EXEC_TYPE)
    used_margin = repo.sum_open_margin_quote(EXEC_TYPE, leverage)

    entry_candidates: list[dict] = []

    for symbol in scan_symbols:
        feat_df = pipeline_result.feature_map.get(symbol)
        if feat_df is None or feat_df.empty:
            continue

        m = _mask_fully_closed(feat_df, main_tf_ms, now_ms)
        closed_df = feat_df.loc[m]
        if closed_df.empty:
            continue
        last_open_ms = int(_series_open_ms(closed_df["timestamp"].iloc[-1:])[0])
        if last_open_ms != btc_last_closed_hour_open_ms:
            continue

        with_barriers = etl.attach_barrier_columns(feat_df.copy())
        row = with_barriers.loc[with_barriers["timestamp"] == closed_df["timestamp"].iloc[-1]].iloc[[0]].copy()
        row["symbol"] = symbol

        missing = [f for f in feature_names if f not in row.columns]
        if missing:
            continue
        nan_feats = [f for f in feature_names if pd.isna(row[f].iloc[0])]
        if nan_feats:
            continue
        if not _is_candidate_event(row, event_filter_config):
            continue
        stop_pct, take_pct = _get_barrier_pcts(row)
        if stop_pct is None or take_pct is None:
            continue

        if _symbol_has_open(repo, symbol):
            continue
        if _cooldown_blocks(repo, symbol, main_tf_ms=main_tf_ms, now_ms=now_ms, btc_signal_open_ms=btc_last_closed_hour_open_ms):
            continue
        if _max_sl_day_blocks(repo, symbol, now_ms):
            continue

        if use_symbol and symbol_categories is not None:
            row["symbol"] = pd.Categorical([symbol], categories=symbol_categories)

        x = _normalize_features_for_model(row, feature_names, symbol_categories)
        x = _apply_feature_clip_bounds(x, clip_bounds)
        proba = model.predict_proba(x)[0]
        p_short, p_long = float(proba[0]), float(proba[1])
        signal, direction_prob, signal_gap = resolve_directional_signal(p_long, p_short)
        if signal == 0:
            continue
        if signal == 1 and not bool(getattr(cfg, "ALLOW_LONGS", True)):
            continue
        if signal == -1 and not bool(getattr(cfg, "ALLOW_SHORTS", True)):
            continue

        entry_hour_start = btc_last_closed_hour_open_ms + main_tf_ms
        m1 = _fetch_closed_1m_range(exchange, symbol, entry_hour_start, entry_hour_start + main_tf_ms, exchange.get_timeframe_ms("1m"))
        now_ms2 = _utc_now_ms()
        tf1 = exchange.get_timeframe_ms("1m")
        m1 = m1.loc[_mask_fully_closed(m1, tf1, now_ms2)].reset_index(drop=True)
        if m1.empty:
            logger.debug("%s: нет закрытой 1m для входа (час %s)", symbol, entry_hour_start)
            continue
        o0 = float(m1.iloc[0]["open"])
        slip = float(getattr(cfg, "SLIPPAGE", 0.0003))
        entry_price = o0 * (1 + slip) if signal == 1 else o0 * (1 - slip)

        risk_capital = balance * float(getattr(cfg, "RISK_PER_TRADE", 0.01))
        position_notional = min(risk_capital / stop_pct, balance * leverage)
        required_margin = position_notional / leverage
        if position_notional < 10 or required_margin <= 0:
            continue

        entry_candidates.append(
            {
                "sym": symbol,
                "signal": signal,
                "entry_price": entry_price,
                "position_notional": position_notional,
                "required_margin": required_margin,
                "stop_pct": stop_pct,
                "take_pct": take_pct,
                "p_long": p_long,
                "p_short": p_short,
                "direction_prob": direction_prob,
                "signal_gap": signal_gap,
                "score": build_entry_score(direction_prob, signal_gap),
                "signal_bar_open_ms": btc_last_closed_hour_open_ms,
                "entry_bar_open_ms": entry_hour_start,
                "entry_ts_ms": int(_series_open_ms(m1["timestamp"].iloc[[0]])[0]),
            }
        )

    entry_candidates.sort(
        key=lambda c: (c["score"], c["direction_prob"]),
        reverse=True,
    )
    max_new = int(getattr(cfg, "BACKTEST_MAX_NEW_POSITIONS_PER_BAR", 1))
    max_open = int(getattr(cfg, "BACKTEST_MAX_OPEN_POSITIONS", 10))
    opened = 0
    open_count = repo.count_open_trades(EXEC_TYPE)

    for c in entry_candidates:
        if opened >= max_new:
            break
        if open_count >= max_open:
            break
        available = balance - used_margin
        if available <= 0:
            break
        req_m = min(c["required_margin"], available)
        notional = min(c["position_notional"], req_m * leverage)
        if notional < 10 or req_m <= 0:
            continue

        scan_open_ms = int(c["entry_ts_ms"])
        last_1m_scan = scan_open_ms

        tid = repo.insert_open_trade(
            execution_type=EXEC_TYPE,
            exchange_code=exchange.get_exchange_code(),
            symbol=c["sym"],
            direction=int(c["signal"]),
            timeframe_signal=main_tf,
            signal_bar_open_ms=int(c["signal_bar_open_ms"]),
            entry_bar_open_ms=int(c["entry_bar_open_ms"]),
            entry_ts_ms=int(c["entry_ts_ms"]),
            entry_price=float(c["entry_price"]),
            entry_notional=float(notional),
            stop_pct=float(c["stop_pct"]),
            take_pct=float(c["take_pct"]),
            p_long=float(c["p_long"]),
            p_short=float(c["p_short"]),
            signal_gap=float(c["signal_gap"]),
            model_name=model_name,
            last_1m_scan_open_ms=last_1m_scan,
            meta={"entry_score": float(c["score"])},
        )
        used_margin += req_m
        open_count += 1
        opened += 1
        logger.info(
            "OPEN id=%s %s %s @ %.8g  notional=%.2f  p_long=%.3f p_short=%.3f",
            tid,
            c["sym"],
            "LONG" if c["signal"] == 1 else "SHORT",
            c["entry_price"],
            notional,
            c["p_long"],
            c["p_short"],
        )


def run_tick(*, verbose: bool = False) -> None:
    _setup_logging(verbose)
    repo = create_trades_repository_from_config()
    repo.init_schema()

    models_dir = Path(getattr(cfg, "MODELS_DIR", "models"))
    model_name = str(getattr(cfg, "PAPER_MODEL_NAME", "lightgbm_target"))
    model_path = models_dir / f"{model_name}.joblib"
    meta_path = models_dir / f"{model_name}_features.json"
    if not model_path.is_file() or not meta_path.is_file():
        logger.error("Нужны %s и %s (train.py)", model_path, meta_path)
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        features_meta = json.load(f)
    feature_names = features_meta["feature_columns"]
    trained_symbols = list(features_meta.get("symbols", []))
    event_filter_meta = features_meta.get("event_filter")
    if event_filter_meta is None:
        logger.error("В JSON нет event_filter — переобучите train.py")
        sys.exit(1)
    event_filter_config = resolve_event_filter_config(event_filter_meta)
    clip_bounds = (features_meta.get("feature_clip") or {}).get("bounds") or {}

    exchange = etl.create_exchange_service()
    clock_sym = _clock_symbol()
    clock_sym = str(exchange.normalize_symbol(clock_sym))
    main_tf = str(getattr(cfg, "TIMEFRAME", "1h"))
    main_tf_ms = exchange.get_timeframe_ms(main_tf)
    tf_1m_ms = exchange.get_timeframe_ms("1m")
    now_ms = _utc_now_ms()

    btc_1h = _fetch_ohlcv_bars(exchange, clock_sym, main_tf, 48)
    btc_1m = _fetch_ohlcv_bars(exchange, clock_sym, "1m", 240)

    last_hour_open = _last_closed_bar_open_ms(btc_1h, main_tf_ms, now_ms)
    last_1m_open = _last_closed_bar_open_ms(btc_1m, tf_1m_ms, now_ms)
    if last_hour_open is None or last_1m_open is None:
        logger.warning("Не удалось определить закрытые бары BTC (час/мин).")
        return

    prev_hour = repo.get_state(STATE_LAST_HOUR)
    prev_hour_i = int(prev_hour) if prev_hour is not None else -1

    scan_symbols = [str(exchange.normalize_symbol(s)) for s in dict.fromkeys(cfg.SYMBOLS)]
    use_symbol = "symbol" in feature_names
    symbol_categories = (
        list(dict.fromkeys(trained_symbols + scan_symbols)) if use_symbol else None
    )
    model = joblib.load(model_path)

    for tr in _load_open_trades_full(repo):
        if _process_minute_exits_for_trade(
            exchange,
            tr,
            last_closed_1m_open_ms=last_1m_open,
            now_ms=now_ms,
            tf_1m_ms=tf_1m_ms,
        ):
            now_ms = _utc_now_ms()

    if last_hour_open > prev_hour_i:
        run_hourly_entries(
            exchange=exchange,
            model=model,
            feature_names=feature_names,
            symbol_categories=symbol_categories,
            event_filter_config=event_filter_config,
            clip_bounds=clip_bounds,
            model_name=model_name,
            repo=repo,
            main_tf=main_tf,
            main_tf_ms=main_tf_ms,
            btc_last_closed_hour_open_ms=last_hour_open,
            now_ms=_utc_now_ms(),
            main_bars=int(getattr(cfg, "PAPER_MAIN_BARS", 3000)),
            htf_bars=int(getattr(cfg, "PAPER_HTF_BARS", 900)),
            min_main_rows=int(getattr(cfg, "PAPER_MIN_MAIN_ROWS", 400)),
            min_htf_rows=int(getattr(cfg, "PAPER_MIN_HTF_ROWS", 120)),
        )
        repo.set_state(STATE_LAST_HOUR, str(last_hour_open))
        logger.info("Часовой тик BTC: обработан бар open_ms=%s", last_hour_open)


def _load_open_trades_full(repo: BaseTradesRepository) -> list[dict]:
    out: list[dict] = []
    for r in repo.list_open_trades(EXEC_TYPE):
        if r.entry_ts_ms is None:
            logger.warning("Пропуск open trade id=%s: отсутствует entry_ts_ms", r.id)
            continue
        out.append(
            {
                "id": int(r.id),
                "symbol": str(r.symbol),
                "direction": int(r.direction),
                "entry_price": float(r.entry_price),
                "entry_notional": float(r.entry_notional),
                "stop_pct": float(r.stop_pct),
                "take_pct": float(r.take_pct),
                "last_1m_scan_open_ms": (
                    int(r.last_1m_scan_open_ms) if r.last_1m_scan_open_ms is not None else None
                ),
                "entry_ts_ms": int(r.entry_ts_ms),
                "repo": repo,
            }
        )
    return out


def cmd_status() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    repo = create_trades_repository_from_config()
    repo.init_schema()
    leverage = float(getattr(cfg, "LEVERAGE", 1))
    initial = float(getattr(cfg, "PAPER_INITIAL_BALANCE", getattr(cfg, "BACKTEST_INITIAL_BALANCE", 100.0)))
    closed_pnl = repo.sum_closed_pnl_quote(EXEC_TYPE)
    margin = repo.sum_open_margin_quote(EXEC_TYPE, leverage)
    wallet = initial + closed_pnl
    available = wallet - margin
    logger.info(
        "paper | wallet=%.2f (initial=%.2f + realized_pnl=%.2f) | margin_open=%.2f | "
        "доступно=%.2f | открыто позиций=%s",
        wallet,
        initial,
        closed_pnl,
        margin,
        available,
        repo.count_open_trades(EXEC_TYPE),
    )
    for t in repo.list_open_trades(EXEC_TYPE):
        logger.info(
            "  OPEN id=%s %s %s entry=%.8g notional=%.2f",
            t.id,
            t.symbol,
            "LONG" if t.direction == 1 else "SHORT",
            t.entry_price,
            t.entry_notional,
        )


def cmd_daemon(verbose: bool) -> None:
    poll = float(getattr(cfg, "PAPER_DAEMON_POLL_SEC", 45.0))
    while True:
        try:
            run_tick(verbose=verbose)
        except KeyboardInterrupt:
            logger.info("Останов по Ctrl+C")
            raise
        except Exception:
            logger.exception("Ошибка тика, повтор через %s с", poll)
        time.sleep(poll)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Пейпер-трейдинг (журнал execution_trades, sync по BTC).",
        epilog=(
            "Без аргументов — непрерывный режим до Ctrl+C (PAPER_DAEMON_POLL_SEC). "
            "Один проход: python paper.py tick."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "command",
        nargs="?",
        default="daemon",
        choices=("tick", "daemon", "status"),
        help="по умолчанию daemon — цикл до остановки; tick — один проход; status — сводка",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.command == "status":
        cmd_status()
        return
    if args.command == "daemon":
        _setup_logging(args.verbose)
        cmd_daemon(args.verbose)
        return
    run_tick(verbose=args.verbose)


if __name__ == "__main__":
    main()