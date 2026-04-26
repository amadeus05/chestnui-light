"""Paper trading mode — реальные данные, фейковое исполнение."""
from __future__ import annotations

import sys

import config
from src.application.executors import SimulationExecutor
from src.application.feeds import LiveDataFeed
from src.application.trading_engine import TradingEngine
from src.contracts import WallClock
from src.domain.execution.execution_service import ExecutionService
from src.domain.execution.models.constraints import BarEntryConstraints
from src.domain.portfolio.portfolio_manager import PortfolioManager
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.models.risk_limits import RiskLimits
from src.domain.signals.signal_brain import SignalBrain
from src.persistence.exchanges.bybit.bybit_service import BybitService


def _attach_feature_provider(signal_brain: SignalBrain) -> None:
    """
    Подключает FeatureProvider к SignalBrain для paper-режима.

    Использует precomputed-фичи из БД. Для свежих баров, которых ещё нет
    в БД, сигналы не генерируются — это ожидаемое поведение при первом
    запуске после обновления данных. Перезапусти train.py для обновления фич.
    """
    from src.application.backtest import backtest_data as ld
    from src.application.backtest.feature_providers import PrecomputedFeatureProvider

    feature_names = signal_brain.get_feature_names()
    if not feature_names:
        print("[run_paper] Модель не загружена — пропускаем feature provider.")
        return

    symbol_categories = signal_brain._symbol_categories
    clip_bounds = signal_brain._clip_bounds
    required_columns = feature_names + ["barrier_stop_pct", "barrier_take_pct"]

    all_features: dict = {}
    for sym in config.SYMBOLS:
        feat_df = ld.load_precomputed_features(
            sym,
            symbol_categories=symbol_categories,
            required_columns=required_columns,
        )
        if feat_df.empty:
            print(f"[run_paper] Нет precomputed-фич для {sym} — символ пропущен.")
            continue
        all_features[sym] = feat_df

    if not all_features:
        print(
            "[run_paper] ⚠️  Precomputed-фичи не найдены. "
            "Сигналы генерироваться не будут. Запусти train.py."
        )
        return

    provider = PrecomputedFeatureProvider(
        all_features_by_symbol=all_features,
        feature_names=feature_names,
        symbol_categories=symbol_categories,
        clip_bounds=clip_bounds or {},
    )
    signal_brain.set_feature_provider(provider)
    print(f"[run_paper] FeatureProvider подключён для {len(all_features)} символ(ов).")


def build_paper_mode() -> TradingEngine:
    """Сборка paper мода — реальные данные, фейковые ордера."""

    # 1. Clock — реальное время
    clock = WallClock()

    # 2. Exchange — только публичные данные (без API ключей)
    exchange = BybitService(
        testnet=True,
    )

    # 3. DataFeed — live данные с биржи
    data_feed = LiveDataFeed(
        clock=clock,
        exchange=exchange,
        poll_interval_sec=getattr(config, "PAPER_DAEMON_POLL_SEC", 45.0),
    )

    # 4. Executor — симулятор (та же логика, что и в бэктесте!)
    executor = SimulationExecutor(
        clock=clock,
        slippage=config.SLIPPAGE,
        taker_fee=config.TAKER_COM,
    )

    # 5. Portfolio — фиктивный баланс
    paper_balance = getattr(config, "PAPER_INITIAL_BALANCE", 10000.0)
    portfolio = PortfolioManager(
        initial_balance=paper_balance,
        taker_com=config.TAKER_COM,
    )

    # 6. Risk — идентичные лимиты
    risk_limits = RiskLimits(
        risk_per_trade=config.RISK_PER_TRADE,
        leverage=config.LEVERAGE,
        min_position_notional=10.0,
        reduced_risk_per_trade=getattr(
            config, "REDUCED_RISK_PER_TRADE", config.RISK_PER_TRADE
        ),
        reduce_risk_after_consecutive_losses=getattr(
            config, "REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0
        ),
    )
    risk_manager = RiskManager(limits=risk_limits)

    # 7. Execution — те же ограничения
    constraints = BarEntryConstraints(
        max_new_positions_per_bar=getattr(config, "MAX_NEW_POSITIONS_PER_BAR", 1),
        max_open_positions=getattr(config, "MAX_OPEN_POSITIONS", 1),
    )
    execution_service = ExecutionService(constraints)

    # 8. Signal Brain — та же модель + feature provider
    signal_brain = SignalBrain(
        model_path=config.MODELS_DIR / f"{config.MODEL_NAME}.joblib",
        features_meta_path=config.MODELS_DIR / f"{config.MODEL_NAME}_features.json",
        threshold=getattr(config, "DIRECTIONAL_PROBA_THRESHOLD", 0.5),
        min_signal_gap=getattr(config, "MIN_SIGNAL_GAP", 0.0),
        allow_longs=getattr(config, "ALLOW_LONGS", True),
        allow_shorts=getattr(config, "ALLOW_SHORTS", True),
    )
    _attach_feature_provider(signal_brain)

    # 9. Engine — тот же класс, что и в бэктесте!
    return TradingEngine(
        clock=clock,
        data_feed=data_feed,
        order_executor=executor,
        portfolio=portfolio,
        risk_manager=risk_manager,
        execution_service=execution_service,
        signal_brain=signal_brain,
        symbols=config.SYMBOLS,
        timeframe=config.TIMEFRAME,
        slippage=config.SLIPPAGE,
        taker_fee=config.TAKER_COM,
        use_open_price_for_entry=True,
        sl_cooldown_bars=getattr(config, "SL_COOLDOWN_BARS", 0),
        max_sl_per_day=getattr(config, "MAX_SL_PER_DAY", 0),
    )


def main() -> int:
    """Точка входа для paper trading."""
    print("=" * 60)
    print("PAPER TRADING MODE")
    print("=" * 60)
    print("⚠️  Используются реальные рыночные данные")
    print("⚠️  Ордера НЕ отправляются на биржу (симуляция)")
    print("=" * 60)

    try:
        engine = build_paper_mode()
        engine.run()
        return 0

    except KeyboardInterrupt:
        print("\n[Paper] Stopped by user")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
