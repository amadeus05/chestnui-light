"""Цикл TradingEngine для портфельного бэктеста (close на decision_ts → исполнение на exec-баре)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.application.backtest import backtest_data as ld
from src.application.backtest.feature_providers import (
    DataFramePredictionProvider,
    PrecomputedFeatureProvider,
    RealtimeFeatureProvider,
)
from src.application.backtest.portfolio_backtest_runner import _build_entry_score_cfg
from src.domain.execution.bar_fill import NextBarOHLC, flatten_at_mark_price, try_stop_take_fill
from src.domain.signals.signal_brain import SignalBrain


def _get_portfolio_signal_brain(engine: "TradingEngine", ctx, cfg) -> SignalBrain:
    """Один экземпляр SignalBrain на прогон портфельного replay (как в portfolio_backtest_runner)."""
    existing = getattr(engine, "_portfolio_signal_brain", None)
    if existing is not None:
        return existing

    brain = SignalBrain(
        threshold=cfg.directional_proba_threshold,
        min_signal_gap=cfg.min_signal_gap,
        allow_longs=cfg.allow_longs,
        allow_shorts=cfg.allow_shorts,
    )
    brain.apply_runtime_config(
        model=None if ctx.using_external_predictions else ctx.model,
        feature_names=ctx.feature_names,
        clip_bounds=ctx.clip_bounds,
        symbol_categories=ctx.symbol_categories,
        event_filter_config=ctx.event_filter_config,
    )
    if cfg.realtime_features:
        brain.set_feature_provider(
            RealtimeFeatureProvider(
                all_raw=ctx.all_raw,
                feature_names=ctx.feature_names,
                symbol_categories=ctx.symbol_categories,
            )
        )
    else:
        brain.set_feature_provider(
            PrecomputedFeatureProvider(
                all_features_by_symbol=ctx.all_features,
                feature_names=ctx.feature_names,
                symbol_categories=ctx.symbol_categories,
                clip_bounds=ctx.clip_bounds,
            )
        )
    if ctx.using_external_predictions and ctx.prediction_lookup:
        brain.set_prediction_provider(DataFramePredictionProvider(ctx.prediction_lookup))

    engine._portfolio_signal_brain = brain
    return brain


def run_portfolio_replay_loop(engine: "TradingEngine") -> None:
    """Полный проход по PortfolioReplayFeed (та же семантика, что portfolio backtest runner)."""
    ctx = engine._portfolio_backtest_context
    assert ctx is not None
    cfg = ctx.run_config
    feed = engine._portfolio_replay_feed
    assert feed is not None

    universe = list(ctx.all_raw.keys())
    engine._portfolio_universe = universe
    engine._portfolio_step_i = -1
    engine._stop_cooldown_until_step = {sym: -1 for sym in universe}

    for sym in universe:
        engine._portfolio.ensure_symbol_slot(sym)

    while True:
        step = feed.next_step()
        if step is None:
            break

        engine._portfolio_step_i += 1
        i = engine._portfolio_step_i

        if hasattr(engine._executor, "set_current_bar"):
            engine._executor.set_current_bar(step.exec_bars)
        engine._executor.sync_with_portfolio(engine._portfolio)

        engine._update_daily_counters(step.exec_ts)

        equity = engine._portfolio.observe_equity(step.mark_prices)
        engine._equity_curve.append((step.decision_ts, equity))

        _process_portfolio_step_exits(engine, step, cfg, i)
        engine._opened_this_bar = 0

        if cfg.max_sl_per_day > 0 and engine._daily_sl_count >= cfg.max_sl_per_day:
            if engine._on_bar:
                engine._on_bar(step.exec_ts, step.exec_bars, equity)
            continue

        _process_portfolio_step_entries(engine, step, ctx, cfg, i)

        if engine._on_bar:
            engine._on_bar(step.exec_ts, step.exec_bars, equity)


def _process_portfolio_step_exits(engine: "TradingEngine", step, cfg, i: int) -> None:
    from src.application.trading_engine import TradeRecord

    for sym in engine._portfolio_universe:
        pos = engine._portfolio.position(sym)
        if pos is None:
            continue
        bar = step.exec_bars.get(sym)
        if bar is None:
            continue
        ohlc = NextBarOHLC(open=bar.open, high=bar.high, low=bar.low)
        fill = try_stop_take_fill(
            int(pos["dir"]),
            float(pos["entry"]),
            float(pos["stop_pct"]),
            float(pos["take_pct"]),
            ohlc,
            cfg.slippage,
        )
        if fill is None:
            continue

        closed = dict(pos)
        pnl_clean, trade_profit, commission = engine._portfolio.close_position_at_price(sym, fill.exit_price)

        side = "SELL" if int(closed["dir"]) == 1 else "BUY"
        try:
            engine._executor.submit_market_order(sym, side, float(closed["size"]))
        except Exception as e:
            print(f"[TradingEngine] Order error: {e}")

        trade = TradeRecord(
            trade_number=int(closed["trade_number"]),
            symbol=sym,
            direction="LONG" if int(closed["dir"]) == 1 else "SHORT",
            entry_time=closed["ts_open"],
            exit_time=step.exec_ts,
            entry_price=float(closed["entry"]),
            exit_price=fill.exit_price,
            size=float(closed["size"]),
            margin=float(closed["margin"]),
            pnl_pct=pnl_clean,
            pnl_abs=trade_profit,
            commission=commission,
            reason=fill.reason,  # type: ignore[arg-type]
            stop_pct=float(closed["stop_pct"]),
            take_pct=float(closed["take_pct"]),
        )
        engine._trades.append(trade)

        if pnl_clean > 0:
            engine._consecutive_losses = 0
        else:
            engine._consecutive_losses += 1

        if fill.reason == "SL":
            if cfg.sl_cooldown_bars > 0:
                engine._stop_cooldown_until_step[sym] = i + cfg.sl_cooldown_bars
            engine._daily_sl_count += 1

        if engine._on_trade:
            engine._on_trade(trade)

        pnl_sign = "+" if pnl_clean > 0 else ""
        icon = "✅" if fill.reason == "TP" else "❌" if fill.reason == "SL" else "⏹"
        print(
            f"[{step.exec_ts}] #{closed['trade_number']} {icon} {sym}: "
            f"{fill.reason} | PnL: {pnl_sign}{pnl_clean * 100:.2f}% | "
            f"Bal: ${engine._portfolio.cash:.2f}"
        )


def _process_portfolio_step_entries(engine: "TradingEngine", step, ctx, cfg, i: int) -> None:
    market_batch: dict[str, dict] = {}
    for sym, bar in step.exec_bars.items():
        if sym not in step.mark_prices:
            continue
        market_batch[sym] = {
            "next_open": float(bar.open),
            "next_high": float(bar.high),
            "next_low": float(bar.low),
            "current_close": float(step.mark_prices[sym]),
        }

    snapshot_balance = engine._portfolio.cash
    entry_candidates: list[dict] = []
    signal_brain = _get_portfolio_signal_brain(engine, ctx, cfg)

    if cfg.realtime_features:
        for sym, ctx_m in market_batch.items():
            if engine._portfolio.position(sym) is not None:
                continue
            if i < engine._stop_cooldown_until_step.get(sym, -1):
                continue
            pred = signal_brain.predict_full(sym, step.exec_ts)
            if pred is None:
                continue
            signal = pred.signal
            stop_pct = pred.stop_pct
            take_pct = pred.take_pct
            p_long = pred.p_long
            p_short = pred.p_short
            direction_prob = pred.direction_prob
            signal_gap = pred.signal_gap

            direction_str = "LONG" if signal == 1 else "SHORT"
            o = ctx_m["next_open"]
            entry_price = o * (1 + cfg.slippage) if signal == 1 else o * (1 - cfg.slippage)
            position_notional, required_margin = engine._risk.raw_entry_notional_and_margin(
                snapshot_balance, stop_pct, engine._consecutive_losses
            )
            if not engine._risk.passes_min_notional(position_notional):
                continue
            entry_candidates.append(
                {
                    "sym": sym,
                    "signal": signal,
                    "direction_str": direction_str,
                    "entry_price": entry_price,
                    "position_notional": position_notional,
                    "required_margin": required_margin,
                    "stop_pct": stop_pct,
                    "take_pct": take_pct,
                    "p_long": p_long,
                    "p_short": p_short,
                    "direction_prob": direction_prob,
                    "score": _build_entry_score_cfg(direction_prob, signal_gap, cfg),
                    "bar": step.exec_bars[sym],
                }
            )
    else:
        for sym, ctx_m in market_batch.items():
            if engine._portfolio.position(sym) is not None:
                continue
            if sym not in ctx.all_features_prepared:
                continue
            if i < engine._stop_cooldown_until_step.get(sym, -1):
                continue
            pred = signal_brain.predict_full(sym, step.decision_ts)
            if pred is None:
                continue
            signal = pred.signal
            stop_pct = pred.stop_pct
            take_pct = pred.take_pct
            p_long = pred.p_long
            p_short = pred.p_short
            direction_prob = pred.direction_prob
            signal_gap = pred.signal_gap

            direction_str = "LONG" if signal == 1 else "SHORT"
            o = ctx_m["next_open"]
            entry_price = o * (1 + cfg.slippage) if signal == 1 else o * (1 - cfg.slippage)
            position_notional, required_margin = engine._risk.raw_entry_notional_and_margin(
                snapshot_balance, stop_pct, engine._consecutive_losses
            )
            if not engine._risk.passes_min_notional(position_notional):
                continue
            entry_candidates.append(
                {
                    "sym": sym,
                    "signal": signal,
                    "direction_str": direction_str,
                    "entry_price": entry_price,
                    "position_notional": position_notional,
                    "required_margin": required_margin,
                    "stop_pct": stop_pct,
                    "take_pct": take_pct,
                    "p_long": p_long,
                    "p_short": p_short,
                    "direction_prob": direction_prob,
                    "score": _build_entry_score_cfg(direction_prob, signal_gap, cfg),
                    "bar": step.exec_bars[sym],
                }
            )

    if not entry_candidates:
        return

    entry_candidates.sort(key=lambda c: (c["score"], c["direction_prob"]), reverse=True)

    opened_this_bar = 0
    open_positions_count = sum(
        1 for s in engine._portfolio_universe if engine._portfolio.position(s) is not None
    )

    for candidate in entry_candidates:
        if not engine._execution.may_open_more(opened_this_bar, open_positions_count):
            break
        available_balance = engine._portfolio.available_balance()
        if available_balance <= 0:
            break
        required_margin = min(candidate["required_margin"], available_balance)
        position_notional = min(candidate["position_notional"], required_margin * cfg.leverage)
        if position_notional < cfg.min_position_notional or required_margin <= 0:
            continue

        trade_number = engine._trade_counter
        engine._trade_counter += 1
        engine._portfolio.open_position(
            candidate["sym"],
            trade_number=trade_number,
            direction=int(candidate["signal"]),
            entry_price=float(candidate["entry_price"]),
            position_notional=float(position_notional),
            required_margin=float(required_margin),
            stop_pct=float(candidate["stop_pct"]),
            take_pct=float(candidate["take_pct"]),
            ts_open=step.exec_ts,
        )
        opened_this_bar += 1
        open_positions_count += 1

        side = "BUY" if int(candidate["signal"]) == 1 else "SELL"
        try:
            engine._executor.submit_market_order(
                candidate["sym"], side, float(position_notional)
            )
        except Exception as e:
            print(f"[TradingEngine] Entry order error: {e}")

        print(
            f"[{step.exec_ts}] #{trade_number} 🔥 OPEN {candidate['direction_str']}: {candidate['sym']} "
            f"(Long={candidate['p_long']:.2f}, Short={candidate['p_short']:.2f}, "
            f"Score={candidate['score']:.3f}) "
            f"at {candidate['entry_price']:.4f} | "
            f"Size: {position_notional:.2f}$ "
            f"Margin: {required_margin:.2f}$"
        )


def finalize_portfolio_backtest(engine: "TradingEngine") -> None:
    from src.application.trading_engine import TradeRecord

    feed = engine._portfolio_replay_feed
    assert feed is not None
    last_mark = feed.final_mark_prices()
    last_ts = feed.last_timestamp()
    if last_ts is not None:
        engine._clock.set(last_ts)

    for sym in engine._portfolio_universe:
        pos = engine._portfolio.position(sym)
        if pos is None:
            continue
        mark_price = last_mark.get(sym)
        if mark_price is None or not np.isfinite(mark_price):
            continue
        closed = dict(pos)
        exit_price = flatten_at_mark_price(int(closed["dir"]), mark_price, engine._slippage)
        pnl_clean, trade_profit, commission = engine._portfolio.close_position_at_price(sym, exit_price)
        side = "SELL" if int(closed["dir"]) == 1 else "BUY"
        try:
            engine._executor.submit_market_order(sym, side, float(closed["size"]))
        except Exception as e:
            print(f"[TradingEngine] Final close error: {e}")

        trade = TradeRecord(
            trade_number=int(closed["trade_number"]),
            symbol=sym,
            direction="LONG" if int(closed["dir"]) == 1 else "SHORT",
            entry_time=closed["ts_open"],
            exit_time=last_ts,
            entry_price=float(closed["entry"]),
            exit_price=exit_price,
            size=float(closed["size"]),
            margin=float(closed["margin"]),
            pnl_pct=pnl_clean,
            pnl_abs=trade_profit,
            commission=commission,
            reason="FINAL",
            stop_pct=float(closed["stop_pct"]),
            take_pct=float(closed["take_pct"]),
        )
        engine._trades.append(trade)
        pnl_sign = "+" if pnl_clean > 0 else ""
        print(
            f"[{last_ts}] #{closed['trade_number']} ⏹ CLOSE {sym}: "
            f"FINAL | PnL: {pnl_sign}{pnl_clean * 100:.2f}% | "
            f"Bal: ${engine._portfolio.cash:.2f}"
        )

    final_equity = engine._portfolio.cash
    if last_ts is not None and last_mark:
        final_equity = engine._portfolio.equity(last_mark)
        engine._equity_curve.append((last_ts, final_equity))

    print(f"\n[TradingEngine] Final balance: ${final_equity:.2f}")
    print(f"[TradingEngine] Total trades: {len(engine._trades)}")
    print("=" * 60)
