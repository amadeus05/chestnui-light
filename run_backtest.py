"""Backtest mode — исторические данные, фейковое исполнение."""
from __future__ import annotations

import sys

import config
from src.application.executors import SimulationExecutor
from src.application.feeds import ReplayDataFeed
from src.application.trading_engine import TradingEngine
from src.contracts import ReplayClock
from src.domain.execution.execution_service import ExecutionService
from src.domain.execution.models.constraints import BarEntryConstraints
from src.domain.portfolio.portfolio_manager import PortfolioManager
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.models.risk_limits import RiskLimits
from src.domain.signals.signal_brain import SignalBrain
from src.persistence.repositories.historical_kline_repo import (
    HistoricalKlineRepository,
)


def _attach_feature_provider(signal_brain: SignalBrain) -> None:
    """
    Загружает precomputed-фичи из БД и подключает PrecomputedFeatureProvider к SignalBrain.
    Если фичи не найдены — SignalBrain не сгенерирует ни одного сигнала.
    """
    from src.application.backtest import backtest_data as ld
    from src.application.backtest.feature_providers import PrecomputedFeatureProvider

    feature_names = signal_brain.get_feature_names()
    if not feature_names:
        print("[run_backtest] Модель не загружена — пропускаем feature provider.")
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
            print(f"[run_backtest] Нет precomputed-фич для {sym} — символ пропущен.")
            continue
        all_features[sym] = feat_df

    if not all_features:
        print(
            "[run_backtest] ⚠️  Precomputed-фичи не найдены ни для одного символа. "
            "Сигналы генерироваться не будут. Запусти train.py для создания фич."
        )
        return

    provider = PrecomputedFeatureProvider(
        all_features_by_symbol=all_features,
        feature_names=feature_names,
        symbol_categories=symbol_categories,
        clip_bounds=clip_bounds or {},
    )
    signal_brain.set_feature_provider(provider)
    print(f"[run_backtest] FeatureProvider подключён для {len(all_features)} символ(ов).")


def build_backtest_mode() -> TradingEngine:
    """Сборка бэктест мода — все компоненты через DI."""

    # 1. Clock — симуляция времени
    clock = ReplayClock()

    # 2. DataFeed — исторические данные из БД
    repository = HistoricalKlineRepository(
        db_path=config.DB_PATH,
        exchange_code=config.EXCHANGE_CODE,
    )
    data_feed = ReplayDataFeed(clock, repository)

    # 3. Executor — симулятор (не ходит на биржу)
    executor = SimulationExecutor(
        clock=clock,
        slippage=config.SLIPPAGE,
        taker_fee=config.TAKER_COM,
    )

    # 4. Portfolio — баланс и позиции
    portfolio = PortfolioManager(
        initial_balance=config.BACKTEST_INITIAL_BALANCE,
        taker_com=config.TAKER_COM,
    )

    # 5. Risk — сайзинг и лимиты
    risk_limits = RiskLimits(
        risk_per_trade=config.RISK_PER_TRADE,
        leverage=config.LEVERAGE,
        min_position_notional=10.0,
        reduced_risk_per_trade=getattr(
            config, "BACKTEST_REDUCED_RISK_PER_TRADE", config.RISK_PER_TRADE
        ),
        reduce_risk_after_consecutive_losses=getattr(
            config, "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0
        ),
    )
    risk_manager = RiskManager(limits=risk_limits)

    # 6. Execution — ограничения на входы
    constraints = BarEntryConstraints(
        max_new_positions_per_bar=getattr(
            config, "BACKTEST_MAX_NEW_POSITIONS_PER_BAR", 1
        ),
        max_open_positions=getattr(config, "BACKTEST_MAX_OPEN_POSITIONS", 1),
    )
    execution_service = ExecutionService(constraints)

    # 7. Signal Brain — ML модель + feature provider
    signal_brain = SignalBrain(
        model_path=config.MODELS_DIR / f"{config.MODEL_NAME}.joblib",
        features_meta_path=config.MODELS_DIR / f"{config.MODEL_NAME}_features.json",
        threshold=getattr(config, "DIRECTIONAL_PROBA_THRESHOLD", 0.5),
        min_signal_gap=getattr(config, "MIN_SIGNAL_GAP", 0.0),
        allow_longs=getattr(config, "ALLOW_LONGS", True),
        allow_shorts=getattr(config, "ALLOW_SHORTS", True),
    )
    _attach_feature_provider(signal_brain)

    # 8. Engine — сборка из идентичных компонентов
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
        sl_cooldown_bars=getattr(config, "BACKTEST_SL_COOLDOWN_BARS", 0),
        max_sl_per_day=getattr(config, "BACKTEST_MAX_SL_PER_DAY", 0),
    )


def main() -> int:
    """Точка входа для бэктеста."""
    print("=" * 60)
    print("BACKTEST MODE")
    print("=" * 60)

    try:
        engine = build_backtest_mode()
        engine.run()

        trades = engine.trades
        if trades:
            wins = sum(1 for t in trades if t.pnl_pct > 0)
            losses = len(trades) - wins
            total_pnl = sum(t.pnl_abs for t in trades)
            win_rate = 100 * wins / len(trades)

            print(f"\n{'='*60}")
            print("BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"Total trades: {len(trades)}")
            print(f"Wins: {wins} | Losses: {losses}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"Final balance: ${engine._portfolio.cash:.2f}")
        else:
            print("\nНет сделок. Проверь наличие precomputed-фич (запусти train.py).")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
