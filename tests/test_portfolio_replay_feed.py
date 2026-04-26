"""PortfolioReplayFeed: синтетические данные без БД."""
from __future__ import annotations

import pandas as pd

from src.application.backtest import backtest_data as ld
from src.application.feeds.portfolio_replay_feed import PortfolioReplayFeed
from src.contracts.clock import ReplayClock


def test_portfolio_replay_feed_two_steps():
    clock = ReplayClock()
    ts0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    ts2 = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")

    main = pd.DataFrame(
        {
            "timestamp": [ts0, ts1, ts2],
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [100.0, 100.0, 100.0],
            "close_time": [ts0 + pd.Timedelta(hours=1), ts1 + pd.Timedelta(hours=1), ts2 + pd.Timedelta(hours=1)],
        }
    )
    all_raw = {"BTC/USDT": {"main": main, "htf": main}}
    idx = ld.build_timestamp_index(main)
    feed = PortfolioReplayFeed(clock, "1h")
    feed.prepare(all_raw=all_raw, all_main_index={"BTC/USDT": idx}, test_timestamps=[ts0, ts1, ts2])

    s1 = feed.next_step()
    assert s1 is not None
    assert s1.decision_ts == ts0
    assert s1.exec_ts == ts1
    assert s1.mark_prices["BTC/USDT"] == 1.02
    assert s1.exec_bars["BTC/USDT"].open == 1.1

    s2 = feed.next_step()
    assert s2 is not None
    assert s2.decision_ts == ts1
    assert s2.exec_ts == ts2

    assert feed.next_step() is None
