from datetime import timezone

import pandas as pd
import pytest

from src.contracts.clock import ReplayClock, WallClock


def test_replay_clock_set_now():
    c = ReplayClock()
    ts = pd.Timestamp("2024-01-15 12:00:00", tz=timezone.utc)
    c.set(ts)
    assert c.now() == ts


def test_replay_clock_requires_set():
    c = ReplayClock()
    with pytest.raises(RuntimeError, match="set"):
        c.now()


def test_wall_clock_is_utc():
    c = WallClock()
    t = c.now()
    assert t.tzinfo is not None
    assert str(t.tzinfo) in ("UTC", "UTC+00:00") or t.tzinfo == timezone.utc
