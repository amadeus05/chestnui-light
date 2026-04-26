"""Live trading mode — реальные данные и реальные ордера."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import config
from src.application.executors import LiveExecutor
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


def get_api_credentials() -> tuple[str, str]:
    """Получает API ключи из переменных окружения."""
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")

    if not api_key or not api_secret:
        print("Error: Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables")
        print("Example:")
        print('  $env:BYBIT_API_KEY = "your_key"')
        print('  $env:BYBIT_API_SECRET = "your_secret"')
        sys.exit(1)

    return api_key, api_secret


def build_live_mode() -> TradingEngine:
    """Сборка live мода — реальные данные и реальные ордера."""

    # 1. Clock — реальное время
    clock = WallClock()

    # 2. Exchange — market data + торговые методы
    # api_key / api_secret пока хранятся, будут нужны когда добавим торговые методы
    api_key, api_secret = get_api_credentials()
    exchange = BybitService(
        api_key=api_key,
        api_secret=api_secret,
        testnet=getattr(config, "LIVE_USE_TESTNET", False),
    )
    # TODO: заменить на TradingClient(api_key, api_secret) когда будет готов
    # торговый клиент с методами place_market_order / get_position

    # 3. DataFeed — те же live данные, что и в paper
    data_feed = LiveDataFeed(
        clock=clock,
        exchange=exchange,
        poll_interval_sec=getattr(config, "LIVE_POLL_INTERVAL_SEC", 5.0),
    )

    # 4. Executor — реальная биржа!
    # Единственное отличие от paper mode
    executor = LiveExecutor(
        exchange=exchange,
        clock=clock,
    )

    # 5. Portfolio — реальный баланс (будет синхронизирован с биржей)
    # Используем малый начальный баланс, реальный запросим у биржи
    portfolio = PortfolioManager(
        initial_balance=0.0,  # Будет обновлен при первой синхронизации
        taker_com=config.TAKER_COM,
    )

    # 6. Risk — те же лимиты
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

    # 8. Signal Brain — та же модель
    signal_brain = SignalBrain(
        model_path=config.MODELS_DIR / f"{config.MODEL_NAME}.joblib",
        features_meta_path=config.MODELS_DIR / f"{config.MODEL_NAME}_features.json",
        threshold=getattr(config, "DIRECTIONAL_PROBA_THRESHOLD", 0.5),
        min_signal_gap=getattr(config, "MIN_SIGNAL_GAP", 0.0),
        allow_longs=getattr(config, "ALLOW_LONGS", True),
        allow_shorts=getattr(config, "ALLOW_SHORTS", True),
    )

    # 9. Engine — тот же класс, что в paper и backtest!
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
        slippage=config.SLIPPAGE,  # Используется для расчета TP/SL
        taker_fee=config.TAKER_COM,
        use_open_price_for_entry=True,
        sl_cooldown_bars=getattr(config, "SL_COOLDOWN_BARS", 0),
        max_sl_per_day=getattr(config, "MAX_SL_PER_DAY", 0),
    )


def main() -> int:
    """Точка входа для live trading."""
    print("=" * 60)
    print("⚠️  LIVE TRADING MODE")
    print("=" * 60)
    print("🚨 РЕАЛЬНЫЕ ДЕНЬГИ НА КОНУ")
    print("=" * 60)
    print("\nПроверьте настройки:")
    print(f"  Exchange: Bybit")
    print(f"  Symbols: {', '.join(config.SYMBOLS)}")
    print(f"  Timeframe: {config.TIMEFRAME}")
    print(f"  Risk per trade: {config.RISK_PER_TRADE*100:.1f}%")
    print(f"  Leverage: {config.LEVERAGE}x")
    print("")

    # Подтверждение
    if not getattr(config, "LIVE_SKIP_CONFIRMATION", False):
        response = input("Продолжить? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted")
            return 1

    try:
        engine = build_live_mode()
        engine.run()  # Бесконечный цикл

        return 0

    except KeyboardInterrupt:
        print("\n[Live] Stopped by user")
        print("⚠️  Проверьте открытые позиции на бирже!")
        return 0

    except Exception as e:
        print(f"\n[Live] Error: {e}")
        print("⚠️  Проверьте открытые позиции на бирже!")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
