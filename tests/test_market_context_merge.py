import numpy as np
import pandas as pd

import etl


def make_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=5, freq="h"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_400],
        }
    )


def test_attach_funding_context_uses_latest_past_value_only():
    base = make_base_frame()
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-12-31 23:30:00",
                    "2025-01-01 01:30:00",
                    "2025-01-01 03:30:00",
                ]
            ),
            "funding_rate": [0.001, 0.002, 0.003],
        }
    )

    result = etl.attach_funding_context(base.sample(frac=1.0, random_state=7), funding)

    assert result["timestamp"].is_monotonic_increasing
    assert result["funding_rate"].tolist() == [0.001, 0.001, 0.002, 0.002, 0.003]


def test_attach_premium_index_context_uses_latest_past_value_only():
    base = make_base_frame()
    premium = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 02:30:00",
                ]
            ),
            "premium_index_close": [1.001, 1.004],
        }
    )

    result = etl.attach_premium_index_context(base, premium)

    assert result["premium_index_close"].tolist() == [1.001, 1.001, 1.001, 1.004, 1.004]


def test_attach_open_interest_context_uses_latest_past_value_only():
    base = make_base_frame()
    open_interest = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 02:00:00",
                    "2025-01-01 04:00:00",
                ]
            ),
            "open_interest": [10_000.0, 12_000.0, 14_000.0],
        }
    )

    result = etl.attach_open_interest_context(base, open_interest)

    pd.testing.assert_series_equal(
        result["open_interest"],
        pd.Series([np.nan, 10_000.0, 12_000.0, 12_000.0, 14_000.0], name="open_interest"),
    )


def test_context_attachers_create_nan_columns_when_context_is_empty():
    base = make_base_frame()

    with_funding = etl.attach_funding_context(base, pd.DataFrame())
    with_premium = etl.attach_premium_index_context(base, pd.DataFrame())
    with_open_interest = etl.attach_open_interest_context(base, pd.DataFrame())

    assert with_funding["funding_rate"].isna().all()
    assert with_premium["premium_index_close"].isna().all()
    assert with_open_interest["open_interest"].isna().all()


def test_context_attachers_do_not_mutate_inputs():
    base = make_base_frame()
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "funding_rate": [0.001],
        }
    )
    base_before = base.copy(deep=True)
    funding_before = funding.copy(deep=True)

    _ = etl.attach_funding_context(base, funding)

    pd.testing.assert_frame_equal(base, base_before)
    pd.testing.assert_frame_equal(funding, funding_before)
