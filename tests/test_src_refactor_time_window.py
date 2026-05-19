import pandas as pd
import pytest

from src_refactor.core.types import TimeWindow


def test_time_window_parses_metadata_period_payload():
    window = TimeWindow.from_payload(
        {"start": "2025-01-01 00:00:00", "end": "2025-01-02 00:00:00"},
        name="train_period",
    )

    assert window.start == pd.Timestamp("2025-01-01 00:00:00")
    assert window.end == pd.Timestamp("2025-01-02 00:00:00")


def test_time_window_mask_is_inclusive():
    window = TimeWindow.from_payload(
        {"start": "2025-01-01 00:00:00", "end": "2025-01-02 00:00:00"}
    )
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31 23:00:00",
                "2025-01-01 00:00:00",
                "2025-01-02 00:00:00",
                "2025-01-02 01:00:00",
            ]
        )
    )

    assert window.mask(timestamps).tolist() == [False, True, True, False]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"start": "bad", "end": "2025-01-02 00:00:00"},
        {"start": "2025-01-03 00:00:00", "end": "2025-01-02 00:00:00"},
    ],
)
def test_time_window_rejects_invalid_metadata_period_payload(payload):
    with pytest.raises(ValueError):
        TimeWindow.from_payload(payload)
