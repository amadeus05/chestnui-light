"""
Портфельный бэктест: сигнал/марк на закрытии t, исполнение на баре t+1.
Использует PortfolioManager, RiskManager, ExecutionService и bar_fill (как TradingEngine).
Данные и фичи — модуль backtest_data (без циклических импортов с backtest_engine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.application.backtest.backtest_config import BacktestRunConfig
from src.application.backtest_reporting import (
    BacktestReporter,
    BacktestMetrics,
    compact_symbol,
    compute_backtest_metrics,
)
from src.domain.execution.bar_fill import (
    NextBarOHLC,
    flatten_at_mark_price,
    try_stop_take_fill,
)
from src.domain.execution.execution_service import ExecutionService
from src.domain.execution.models.constraints import BarEntryConstraints
from src.domain.portfolio.models.pnl import update_drawdown_stats
from src.domain.portfolio.portfolio_manager import PortfolioManager
from src.domain.risk.models.risk_limits import RiskLimits
from src.domain.risk.risk_manager import RiskManager
from src.domain.signals.signal_brain import SignalBrain

from src.application.backtest import backtest_data as ld
from src.application.backtest.feature_providers import (
    DataFramePredictionProvider,
    PrecomputedFeatureProvider,
    RealtimeFeatureProvider,
)


def _build_entry_score_cfg(direction_prob: float, signal_gap: float, cfg: BacktestRunConfig) -> float:
    edge = max(0.0, direction_prob - cfg.directional_proba_threshold)
    return edge * 10 + signal_gap


def run_portfolio_backtest(
    *,
    model_name: str | None = None,
    model: Any | None = None,
    features_meta: dict | None = None,
    predictions: pd.DataFrame | None = None,
    equity_curve_path: Path | None = None,
    result_title: str = "PORTFOLIO BACKTEST RESULTS",
    run_config: BacktestRunConfig | None = None,
) -> BacktestMetrics | None:
    """
    Полный аналог backtest_engine.backtest(), но состояние счёта ведёт PortfolioManager
    и выходы считаются через try_stop_take_fill.
    """
    print("Loading model and features...")
    cfg = run_config or BacktestRunConfig.from_config_module()
    import config as app_cfg

    if not cfg.allow_longs and not cfg.allow_shorts:
        print("Error: both ALLOW_LONGS and ALLOW_SHORTS are disabled.")
        return None

    using_external_predictions = predictions is not None
    prediction_lookup = ld.build_prediction_lookup(predictions)
    if using_external_predictions and not prediction_lookup:
        print("Error: walk-forward predictions are empty.")
        return None
    if using_external_predictions and cfg.realtime_features:
        print("Error: walk-forward prediction backtest requires BACKTEST_REALTIME_FEATURES=False.")
        return None

    DEFAULT_MODEL_NAME = ld.DEFAULT_MODEL_NAME
    MODELS_DIR = getattr(app_cfg, "MODELS_DIR", Path("models"))

    if features_meta is None:
        import joblib
        import json

        name = model_name or DEFAULT_MODEL_NAME
        model_path = MODELS_DIR / f"{name}.joblib"
        features_meta_path = MODELS_DIR / f"{name}_features.json"
        if not features_meta_path.exists():
            print(f"Error: features metadata not found at {features_meta_path}. Run train.py first.")
            return None
        with open(features_meta_path, "r", encoding="utf-8") as f:
            features_meta = json.load(f)
        if model is None and not using_external_predictions:
            if not model_path.exists():
                print(f"Error: model not found at {model_path}. Run train.py first.")
                return None
            model = joblib.load(model_path)
    elif model is None and not using_external_predictions:
        print("Error: model must be provided when features_meta is passed without predictions.")
        return None

    feature_names = features_meta["feature_columns"]
    try:
        test_start_ts, test_end_ts = ld.parse_period_payload(
            features_meta.get("train_period") or features_meta.get("test_period"),
            "train_period",
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return None

    event_filter_meta = features_meta.get("event_filter")
    if event_filter_meta is None:
        print("Error: model metadata does not include event_filter. Re-run train.py first.")
        return None
    from signal_filter import resolve_event_filter_config

    event_filter_config = resolve_event_filter_config(event_filter_meta)

    trained_symbols = list(features_meta.get("symbols", cfg.symbols))
    use_symbol_feature = "symbol" in feature_names
    unseen_symbols = [s for s in cfg.symbols if s not in trained_symbols]
    if use_symbol_feature:
        symbol_categories = list(dict.fromkeys(trained_symbols + list(cfg.symbols)))
        if unseen_symbols:
            print(
                "Warning: backtest includes symbols absent from training metadata: "
                + ", ".join(unseen_symbols)
            )
    else:
        symbol_categories = None

    feature_clip_meta = features_meta.get("feature_clip", {})
    clip_bounds = feature_clip_meta.get("bounds", {})

    if using_external_predictions:
        print(f"Loaded walk-forward OOS predictions: {len(prediction_lookup)} symbol/timestamp rows")
    else:
        print(f"Loaded LightGBM model with {len(feature_names)} features")
    if clip_bounds:
        print(
            "Feature clipping: "
            f"{len(clip_bounds)} columns "
            f"[{feature_clip_meta.get('lower_q', 0.01) * 100:.2f}%, "
            f"{feature_clip_meta.get('upper_q', 0.99) * 100:.2f}%]"
        )
    print(f"Backtest window: {test_start_ts.isoformat()} to {test_end_ts.isoformat()}")
    if event_filter_config.get("enabled", False):
        print(
            "Event filter: "
            f"|ema_fast_slow|>={event_filter_config.get('min_abs_ema_fast_slow', 0.0):.4f}, "
            f"adx_4h>={event_filter_config.get('min_adx_4h', 0.0):.1f}, "
            f"realized_vol_1h in [{event_filter_config.get('min_realized_vol_1h', 0.0):.4f}, "
            f"{event_filter_config.get('max_realized_vol_1h', 1.0):.4f}]"
        )

    all_raw = ld.load_all_raw_data(cfg.symbols)
    if not all_raw:
        print("Error: no raw data for backtest.")
        return None

    print("Feature mode: " + ("realtime rebuild" if cfg.realtime_features else "precomputed DB features"))

    all_features: dict[str, pd.DataFrame] = {}
    all_main_index: dict[str, dict] = {}
    if not cfg.realtime_features:
        for sym in list(all_raw.keys()):
            feat_df = ld.load_precomputed_features(
                sym,
                symbol_categories=symbol_categories,
                required_columns=feature_names + ["barrier_stop_pct", "barrier_take_pct"],
            )
            if feat_df.empty:
                print(f"Warning: {sym} has no precomputed features table.")
                continue
            all_features[sym] = feat_df

        all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features}
        if not all_raw:
            print("Error: no symbols with precomputed features available.")
            return None

    for sym, payload in all_raw.items():
        all_main_index[sym] = ld.build_timestamp_index(payload["main"])

    all_features_prepared: dict[str, pd.DataFrame] = {}
    if not cfg.realtime_features:
        for sym, feat_df in all_features.items():
            prepared = ld.prepare_precomputed_feature_store(
                feat_df,
                feature_names,
                symbol_categories=symbol_categories,
                clip_bounds=clip_bounds,
            )
            if prepared.empty:
                print(f"Warning: {sym} has no usable precomputed feature rows after preparation.")
                continue
            all_features_prepared[sym] = prepared

        all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features_prepared}
        if not all_raw:
            print("Error: no symbols with prepared precomputed features available.")
            return None

    all_raw, dropped_symbols = ld.filter_symbols_with_period_overlap(
        all_raw, test_start_ts, test_end_ts, min_candles=2
    )
    if dropped_symbols:
        print(
            "Warning: dropped symbols without enough overlap inside the backtest window: "
            + ", ".join(dropped_symbols)
        )
    if not all_raw:
        print("Error: no symbols with enough overlap inside the backtest window.")
        return None

    common_timestamps = ld.get_common_main_timestamps(all_raw)
    test_timestamps = [ts for ts in common_timestamps if test_start_ts <= ts <= test_end_ts]
    if len(test_timestamps) < 2:
        print("Error: too little common data inside the backtest period.")
        return None

    portfolio = PortfolioManager(cfg.initial_balance, taker_com=cfg.taker_com)
    risk = RiskManager(
        RiskLimits(
            risk_per_trade=cfg.risk_per_trade,
            reduced_risk_per_trade=cfg.reduced_risk_per_trade,
            reduce_risk_after_consecutive_losses=cfg.reduce_risk_after_consecutive_losses,
            leverage=cfg.leverage,
            min_position_notional=cfg.min_position_notional,
        )
    )
    execution = ExecutionService(
        BarEntryConstraints(
            max_new_positions_per_bar=cfg.max_new_positions_per_bar,
            max_open_positions=cfg.max_open_positions,
        )
    )

    for sym in all_raw:
        portfolio.ensure_symbol_slot(sym)

    signal_brain = SignalBrain(
        threshold=cfg.directional_proba_threshold,
        min_signal_gap=cfg.min_signal_gap,
        allow_longs=cfg.allow_longs,
        allow_shorts=cfg.allow_shorts,
    )
    signal_brain.apply_runtime_config(
        model=None if using_external_predictions else model,
        feature_names=feature_names,
        clip_bounds=clip_bounds,
        symbol_categories=symbol_categories,
        event_filter_config=event_filter_config,
    )
    if cfg.realtime_features:
        signal_brain.set_feature_provider(
            RealtimeFeatureProvider(
                all_raw=all_raw,
                feature_names=feature_names,
                symbol_categories=symbol_categories,
            )
        )
    else:
        signal_brain.set_feature_provider(
            PrecomputedFeatureProvider(
                all_features_by_symbol=all_features,
                feature_names=feature_names,
                symbol_categories=symbol_categories,
                clip_bounds=clip_bounds,
            )
        )
    if using_external_predictions:
        signal_brain.set_prediction_provider(DataFramePredictionProvider(prediction_lookup))

    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_timestamps: list = []
    monthly_stats: dict[str, dict] = {}
    peak_equity = float(cfg.initial_balance)
    max_drawdown = 0.0
    next_trade_number = 1
    stop_cooldown_until_index = {sym: -1 for sym in all_raw}
    current_trade_day: pd.Timestamp | None = None
    daily_sl_count = 0
    daily_stop_announced = False
    consecutive_loss_count = 0

    initial_balance = cfg.initial_balance

    print("\n📋 Backtest Configuration (PortfolioManager + bar_fill):")
    print(f"   Period: {test_timestamps[0].isoformat()} to {test_timestamps[-1].isoformat()}")
    print(f"   Symbols: {', '.join(compact_symbol(sym) for sym in all_raw.keys())}")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Risk per Trade: {cfg.risk_per_trade * 100:.0f}%")
    print(f"   Leverage: {cfg.leverage:.0f}x")
    print(f"   Main TF: {cfg.timeframe} | HTF: {cfg.htf_timeframe}")
    print(f"   Direction mode: {'LONG+SHORT' if cfg.allow_longs and cfg.allow_shorts else 'LONG ONLY' if cfg.allow_longs else 'SHORT ONLY'}")
    print(f"   Directional probability threshold: {cfg.directional_proba_threshold:.2f}")
    print(f"   Min signal gap: {cfg.min_signal_gap:.2f}")
    print(f"   Batch entries per bar: {cfg.max_new_positions_per_bar}")
    print(f"   Max open positions: {cfg.max_open_positions}")
    print(f"   SL cooldown bars: {cfg.sl_cooldown_bars}")
    print(f"   Max SL per day: {cfg.max_sl_per_day}")
    print(
        "   Reduced risk after consecutive losses: "
        f"{cfg.reduce_risk_after_consecutive_losses} -> {cfg.reduced_risk_per_trade * 100:.2f}%"
    )
    print(f"\nBacktest on {len(test_timestamps)} candles")
    print("-" * 80)

    num_candles = len(test_timestamps)

    for i in range(num_candles - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i + 1]
        trade_day = next_ts.normalize()
        if current_trade_day is None or trade_day != current_trade_day:
            current_trade_day = trade_day
            daily_sl_count = 0
            daily_stop_announced = False

        month_key = next_ts.strftime("%Y-%m")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {
                "pnl_abs": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "start_balance": portfolio.cash,
            }

        market_batch: dict[str, dict] = {}
        for sym, payload in all_raw.items():
            curr_exec = ld.get_exec_row_by_ts_index(all_main_index[sym], current_ts)
            next_exec = ld.get_exec_row_by_ts_index(all_main_index[sym], next_ts)
            if curr_exec is None or next_exec is None:
                continue
            market_batch[sym] = {
                "main": payload["main"],
                "htf": payload["htf"],
                "current_close": float(curr_exec["close"]),
                "next_open": float(next_exec["open"]),
                "next_high": float(next_exec["high"]),
                "next_low": float(next_exec["low"]),
            }

        current_mark_prices = {sym: ctx["current_close"] for sym, ctx in market_batch.items()}
        current_equity = portfolio.equity(current_mark_prices)
        equity_curve.append(current_equity)
        equity_timestamps.append(current_ts)
        peak_equity, max_drawdown = update_drawdown_stats(current_equity, peak_equity, max_drawdown)

        for sym, ctx in market_batch.items():
            pos = portfolio.position(sym)
            if pos is None:
                continue

            ohlc = NextBarOHLC(
                open=ctx["next_open"],
                high=ctx["next_high"],
                low=ctx["next_low"],
            )
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
            pnl_clean, trade_profit, commission = portfolio.close_position_at_price(sym, fill.exit_price)
            previous_loss_streak = consecutive_loss_count

            trades.append(
                {
                    "trade_number": closed["trade_number"],
                    "sym": sym,
                    "direction": "LONG" if int(closed["dir"]) == 1 else "SHORT",
                    "reason": fill.reason,
                    "pnl_pct": pnl_clean,
                    "pnl_abs": trade_profit,
                    "commission": commission,
                    "ts": next_ts,
                }
            )
            monthly_stats[month_key]["pnl_abs"] += trade_profit
            monthly_stats[month_key]["trades"] += 1
            if pnl_clean > 0:
                monthly_stats[month_key]["wins"] += 1
                consecutive_loss_count = 0
            else:
                monthly_stats[month_key]["losses"] += 1
                consecutive_loss_count += 1

            if fill.reason == "SL":
                if cfg.sl_cooldown_bars > 0:
                    stop_cooldown_until_index[sym] = i + cfg.sl_cooldown_bars
                daily_sl_count += 1

            exit_icon = "\u274c" if fill.reason == "SL" else "\u2705" if fill.reason == "TP" else "\u2139\ufe0f"
            print(
                f"[{next_ts}] № {closed['trade_number']} {exit_icon} {sym}: {fill.reason} | "
                f"PnL: {pnl_clean * 100:+.2f}% | "
                f"Com: {commission:.2f}$ | "
                f"Bal: {portfolio.cash:.2f}"
            )
            if (
                cfg.reduce_risk_after_consecutive_losses > 0
                and cfg.reduced_risk_per_trade < cfg.risk_per_trade
            ):
                if (
                    previous_loss_streak < cfg.reduce_risk_after_consecutive_losses
                    <= consecutive_loss_count
                ):
                    print(
                        f"[{next_ts}] ⚠️ Loss streak {consecutive_loss_count}: "
                        f"risk per trade reduced to {cfg.reduced_risk_per_trade * 100:.2f}%"
                    )
                elif (
                    pnl_clean > 0
                    and previous_loss_streak >= cfg.reduce_risk_after_consecutive_losses
                ):
                    print(
                        f"[{next_ts}] ℹ️ Loss streak reset: "
                        f"risk per trade restored to {cfg.risk_per_trade * 100:.2f}%"
                    )
            if (
                fill.reason == "SL"
                and cfg.max_sl_per_day > 0
                and daily_sl_count >= cfg.max_sl_per_day
                and not daily_stop_announced
            ):
                daily_stop_announced = True
                print(
                    f"[{next_ts}] ⛔ Daily SL limit reached ({daily_sl_count}), "
                    "new entries are paused until next day"
                )

        if cfg.max_sl_per_day > 0 and daily_sl_count >= cfg.max_sl_per_day:
            continue

        snapshot_balance = portfolio.cash
        entry_candidates: list[dict] = []

        if cfg.realtime_features:
            for sym, ctx in market_batch.items():
                if portfolio.position(sym) is not None:
                    continue
                if i < stop_cooldown_until_index.get(sym, -1):
                    continue

                pred = signal_brain.predict_full(sym, next_ts)
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
                entry_price = ctx["next_open"] * (1 + cfg.slippage) if signal == 1 else ctx["next_open"] * (1 - cfg.slippage)
                position_notional, required_margin = risk.raw_entry_notional_and_margin(
                    snapshot_balance, stop_pct, consecutive_loss_count
                )
                if not risk.passes_min_notional(position_notional):
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
                    }
                )
        else:
            for sym, ctx in market_batch.items():
                if portfolio.position(sym) is not None:
                    continue
                if sym not in all_features_prepared:
                    continue
                if i < stop_cooldown_until_index.get(sym, -1):
                    continue

                pred = signal_brain.predict_full(sym, current_ts)
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
                entry_price = ctx["next_open"] * (1 + cfg.slippage) if signal == 1 else ctx["next_open"] * (1 - cfg.slippage)
                position_notional, required_margin = risk.raw_entry_notional_and_margin(
                    snapshot_balance, stop_pct, consecutive_loss_count
                )
                if not risk.passes_min_notional(position_notional):
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
                    }
                )

        if not entry_candidates:
            continue

        entry_candidates.sort(
            key=lambda c: (c["score"], c["direction_prob"]),
            reverse=True,
        )

        opened_this_bar = 0
        open_positions_count = sum(1 for s in all_raw if portfolio.position(s) is not None)

        for candidate in entry_candidates:
            if not execution.may_open_more(opened_this_bar, open_positions_count):
                break
            available_balance = portfolio.available_balance()
            if available_balance <= 0:
                break
            required_margin = min(candidate["required_margin"], available_balance)
            position_notional = min(candidate["position_notional"], required_margin * cfg.leverage)
            if position_notional < cfg.min_position_notional or required_margin <= 0:
                continue

            trade_number = next_trade_number
            next_trade_number += 1
            portfolio.open_position(
                candidate["sym"],
                trade_number=trade_number,
                direction=int(candidate["signal"]),
                entry_price=float(candidate["entry_price"]),
                position_notional=float(position_notional),
                required_margin=float(required_margin),
                stop_pct=float(candidate["stop_pct"]),
                take_pct=float(candidate["take_pct"]),
                ts_open=next_ts,
            )
            opened_this_bar += 1
            open_positions_count += 1

            print(
                f"[{next_ts}] № {trade_number} 🔥 OPEN {candidate['direction_str']}: {candidate['sym']} "
                f"(Long={candidate['p_long']:.2f}, Short={candidate['p_short']:.2f}, "
                f"Score={candidate['score']:.3f}) "
                f"at {candidate['entry_price']:.4f} | "
                f"Size: {position_notional:.2f}$ "
                f"Margin: {required_margin:.2f}$"
            )

    last_timestamp = test_timestamps[-1]
    last_mark_prices: dict[str, float] = {}
    for sym in all_raw:
        last_exec = ld.get_exec_row_by_ts_index(all_main_index[sym], last_timestamp)
        if last_exec is None:
            continue
        last_mark_prices[sym] = float(last_exec["close"])

    final_month_key = last_timestamp.strftime("%Y-%m")
    if final_month_key not in monthly_stats:
        monthly_stats[final_month_key] = {
            "pnl_abs": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "start_balance": portfolio.cash,
        }

    for sym in list(all_raw.keys()):
        pos = portfolio.position(sym)
        if pos is None:
            continue
        mark_price = last_mark_prices.get(sym)
        if mark_price is None or not np.isfinite(mark_price):
            continue
        exit_price = flatten_at_mark_price(int(pos["dir"]), mark_price, cfg.slippage)
        pnl_clean, trade_profit, commission = portfolio.close_position_at_price(sym, exit_price)
        trades.append(
            {
                "trade_number": pos["trade_number"],
                "sym": sym,
                "direction": "LONG" if int(pos["dir"]) == 1 else "SHORT",
                "reason": "FINAL",
                "pnl_pct": pnl_clean,
                "pnl_abs": trade_profit,
                "commission": commission,
                "ts": last_timestamp,
            }
        )
        monthly_stats[final_month_key]["pnl_abs"] += trade_profit
        monthly_stats[final_month_key]["trades"] += 1
        if pnl_clean > 0:
            monthly_stats[final_month_key]["wins"] += 1
        else:
            monthly_stats[final_month_key]["losses"] += 1
        print(
            f"[{last_timestamp}] № {pos['trade_number']} ⏹ CLOSE {sym}: FINAL | "
            f"PnL: {pnl_clean * 100:+.2f}% | "
            f"Com: {commission:.2f}$ | "
            f"Bal: {portfolio.cash:.2f}"
        )

    final_equity = portfolio.equity(last_mark_prices)
    equity_curve.append(final_equity)
    equity_timestamps.append(last_timestamp)
    peak_equity, max_drawdown = update_drawdown_stats(final_equity, peak_equity, max_drawdown)

    metrics = compute_backtest_metrics(
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
        trades=trades,
        max_drawdown=max_drawdown,
        final_balance=portfolio.cash,
        initial_balance=initial_balance,
    )
    reporter = BacktestReporter(
        result_title=result_title,
        default_equity_chart_path=cfg.default_equity_chart_path,
        equity_curve_path=equity_curve_path,
    )
    reporter.print_full_console(
        metrics,
        initial_balance=initial_balance,
        monthly_stats=monthly_stats,
        trades=trades,
    )
    reporter.save_equity_chart(
        equity_timestamps=equity_timestamps,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
        total_trades=metrics.total_trades,
        max_drawdown=metrics.max_drawdown,
    )
    return metrics
