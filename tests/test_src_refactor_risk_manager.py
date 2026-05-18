import pandas as pd

from src_refactor.domain.risk.risk_manager import RiskConfig, RiskManager


def test_risk_manager_resets_daily_sl_limit_on_new_trade_day():
    risk = RiskManager(RiskConfig(max_sl_per_day=1))

    risk.on_bar(pd.Timestamp("2025-01-01 01:00:00"))
    risk.record_trade_close(symbol="BTC/USDT", pnl_pct=-0.02, reason="SL", bar_index=3)

    assert risk.state.daily_sl_count == 1
    assert not risk.can_open_symbol("ETH/USDT", bar_index=4, open_positions_count=0)

    risk.on_bar(pd.Timestamp("2025-01-02 00:00:00"))

    assert risk.state.daily_sl_count == 0
    assert risk.can_open_symbol("ETH/USDT", bar_index=5, open_positions_count=0)


def test_risk_manager_keeps_intraday_sl_limit_until_day_changes():
    risk = RiskManager(RiskConfig(max_sl_per_day=1))

    risk.on_bar(pd.Timestamp("2025-01-01 01:00:00"))
    risk.record_trade_close(symbol="BTC/USDT", pnl_pct=-0.02, reason="SL", bar_index=3)
    risk.on_bar(pd.Timestamp("2025-01-01 23:00:00"))

    assert risk.state.daily_sl_count == 1
    assert not risk.can_open_symbol("ETH/USDT", bar_index=4, open_positions_count=0)
