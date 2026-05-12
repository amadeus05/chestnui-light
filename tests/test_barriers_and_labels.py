import numpy as np
import pandas as pd
import pytest

import config as cfg
import etl
from src.features import indicators


def make_label_frame(rows: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.5),
            "low": np.full(rows, 99.5),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1_000.0),
            "realized_vol_1h": np.full(rows, 0.01),
            "barrier_stop_pct": np.full(rows, 0.02),
            "barrier_take_pct": np.full(rows, 0.03),
        }
    )


@pytest.fixture()
def fallback_atr(monkeypatch):
    monkeypatch.setattr(indicators, "_load_pandas_ta", lambda: None)


@pytest.fixture()
def fixed_labeling_config(monkeypatch):
    monkeypatch.setattr(indicators, "_load_pandas_ta", lambda: None)
    monkeypatch.setattr(cfg, "HORIZON", 3)
    monkeypatch.setattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)
    monkeypatch.setattr(cfg, "USE_DYNAMIC_BARRIERS", True)
    monkeypatch.setattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25)
    monkeypatch.setattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)
    monkeypatch.setattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0)
    monkeypatch.setattr(cfg, "BARRIER_MIN_PCT", 0.01)
    monkeypatch.setattr(cfg, "BARRIER_MAX_PCT", 0.05)
    monkeypatch.setattr(cfg, "SL_PCT", 0.015)
    monkeypatch.setattr(cfg, "TP_PCT", 0.03)
    monkeypatch.setattr(cfg, "SLIPPAGE", 0.0)
    monkeypatch.setattr(cfg, "TAKER_COM", 0.0)


def test_effective_horizons_fixed_mode(monkeypatch):
    monkeypatch.setattr(cfg, "HORIZON", 5)
    monkeypatch.setattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)

    horizons = etl.compute_effective_horizons(pd.DataFrame(index=range(4)))

    assert horizons.dtype == np.int32
    assert horizons.tolist() == [5, 5, 5, 5]


def test_effective_horizons_adaptive_mode(monkeypatch):
    monkeypatch.setattr(cfg, "HORIZON", 12)
    monkeypatch.setattr(cfg, "ENABLE_ADAPTIVE_HORIZON", True)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_MIN", 8)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_MAX", 20)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.005)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.025)
    frame = pd.DataFrame(
        {
            "realized_vol_1h": [
                0.001,
                0.005,
                0.015,
                0.025,
                0.100,
                np.nan,
            ]
        }
    )

    horizons = etl.compute_effective_horizons(frame)

    assert horizons.tolist() == [20, 20, 14, 8, 8, 12]


def test_effective_horizons_falls_back_on_invalid_adaptive_bounds(monkeypatch):
    monkeypatch.setattr(cfg, "HORIZON", 7)
    monkeypatch.setattr(cfg, "ENABLE_ADAPTIVE_HORIZON", True)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.02)
    monkeypatch.setattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.02)
    frame = pd.DataFrame({"realized_vol_1h": [0.01, 0.02, 0.03]})

    horizons = etl.compute_effective_horizons(frame)

    assert horizons.tolist() == [7, 7, 7]


def test_attach_barrier_columns_dynamic_mode_clips_and_derives_take_pct(fixed_labeling_config):
    frame = make_label_frame(rows=30).drop(columns=["barrier_stop_pct", "barrier_take_pct"])
    frame["realized_vol_1h"] = np.linspace(0.001, 0.20, len(frame))

    result = etl.attach_barrier_columns(frame)

    assert "barrier_stop_pct" not in frame.columns
    assert "barrier_take_pct" not in frame.columns
    assert result["barrier_stop_pct"].between(0.01, 0.05).all()
    pd.testing.assert_series_equal(
        result["barrier_take_pct"],
        result["barrier_stop_pct"] * 2.0,
        check_names=False,
    )


def test_attach_barrier_columns_static_mode_uses_legacy_config(monkeypatch, fallback_atr):
    monkeypatch.setattr(cfg, "USE_DYNAMIC_BARRIERS", False)
    monkeypatch.setattr(cfg, "SL_PCT", 0.012)
    monkeypatch.setattr(cfg, "TP_PCT", 0.034)
    frame = make_label_frame(rows=5).drop(columns=["barrier_stop_pct", "barrier_take_pct"])

    result = etl.attach_barrier_columns(frame)

    assert result["barrier_stop_pct"].tolist() == [0.012] * 5
    assert result["barrier_take_pct"].tolist() == [0.034] * 5


def test_attach_barrier_columns_dynamic_mode_requires_realized_vol(monkeypatch, fallback_atr):
    monkeypatch.setattr(cfg, "USE_DYNAMIC_BARRIERS", True)
    frame = make_label_frame(rows=5).drop(
        columns=["realized_vol_1h", "barrier_stop_pct", "barrier_take_pct"]
    )

    with pytest.raises(ValueError, match="realized_vol_1h"):
        etl.attach_barrier_columns(frame)


@pytest.mark.parametrize(
    ("highs", "lows", "expected_label"),
    [
        ([100.5, 103.5, 100.5, 100.5, 100.5, 100.5], [99.5, 99.5, 99.5, 99.5, 99.5, 99.5], 1),
        ([100.5, 101.5, 100.5, 100.5, 100.5, 100.5], [99.5, 96.5, 99.5, 99.5, 99.5, 99.5], -1),
        ([100.5, 100.5, 100.5, 100.5, 100.5, 100.5], [99.5, 99.5, 99.5, 99.5, 99.5, 99.5], 0),
    ],
)
def test_triple_barrier_labeling_manual_first_row_scenarios(
    fixed_labeling_config,
    highs,
    lows,
    expected_label,
):
    frame = make_label_frame(rows=6)
    frame["high"] = highs
    frame["low"] = lows

    labeled = etl.triple_barrier_labeling(frame)

    assert labeled.loc[0, "Target"] == expected_label
    assert labeled["Target"].tail(3).tolist() == [0, 0, 0]


def test_triple_barrier_labeling_prefers_neutral_when_both_sides_win(fixed_labeling_config):
    frame = make_label_frame(rows=6)
    frame.loc[1, "high"] = 103.5
    frame.loc[1, "low"] = 96.5

    labeled = etl.triple_barrier_labeling(frame)

    assert labeled.loc[0, "Target"] == 0


def test_triple_barrier_labeling_uses_row_specific_barriers(fixed_labeling_config):
    frame = make_label_frame(rows=6)
    frame.loc[0, "barrier_take_pct"] = 0.05
    frame.loc[1, "high"] = 103.5

    labeled = etl.triple_barrier_labeling(frame)

    assert labeled.loc[0, "Target"] == 0
