from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from lstm.dataset import SequenceDataset, SequenceStandardizer, build_history_by_symbol
from lstm.train_lstm_walk_forward import split_train_eval_indices as split_lstm_train_eval_indices
from lstm.train_lstm_walk_forward import build_features_meta as build_lstm_features_meta
from lstm_candles.train_lstm_candles_walk_forward import (
    build_features_meta as build_candle_lstm_features_meta,
)
from lstm_candles.train_lstm_candles_walk_forward import (
    split_train_eval_indices as split_candle_train_eval_indices,
)

TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
TARGET_COLUMN = "Target"


def make_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2025-01-01 02:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                ]
            ),
            SYMBOL_COLUMN: [
                "BTC/USDT",
                "BTC/USDT",
                "BTC/USDT",
                "BTC/USDT",
                "ETH/USDT",
                "ETH/USDT",
                "ETH/USDT",
            ],
            "feature_a": [3.0, 1.0, 2.0, 20.0, 10.0, 11.0, 12.0],
            "feature_b": [30.0, np.nan, np.inf, 200.0, 100.0, 110.0, 120.0],
        }
    )


def make_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                    "2025-01-01 02:00:00",
                ]
            ),
            SYMBOL_COLUMN: ["BTC/USDT", "BTC/USDT", "BTC/USDT", "ETH/USDT"],
            TARGET_COLUMN: [0, 1, 0, 1],
        }
    )


def test_build_history_by_symbol_sorts_deduplicates_and_sanitizes_infinities():
    history = build_history_by_symbol(make_history_frame(), ["feature_a", "feature_b"])

    assert set(history) == {"BTC/USDT", "ETH/USDT"}
    btc = history["BTC/USDT"]
    assert btc[TIMESTAMP_COLUMN].tolist() == list(pd.date_range("2025-01-01", periods=3, freq="h"))
    assert btc["feature_a"].tolist() == [1.0, 20.0, 3.0]
    assert np.isnan(btc.loc[0, "feature_b"])
    assert btc.loc[1, "feature_b"] == 200.0


def test_sequence_dataset_uses_current_and_past_rows_only():
    history = build_history_by_symbol(make_history_frame(), ["feature_a", "feature_b"])
    dataset = SequenceDataset(
        sample_frame=make_sample_frame(),
        history_by_symbol=history,
        feature_columns=["feature_a", "feature_b"],
        sequence_length=2,
    )

    assert len(dataset) == 3
    assert [sample.row_index for sample in dataset.samples] == [1, 2, 3]

    sequence, target = dataset[0]
    np.testing.assert_allclose(
        sequence.numpy(),
        np.array([[1.0, 0.0], [20.0, 200.0]], dtype=np.float32),
    )
    assert int(target.item()) == 1

    metadata = dataset.metadata_frame()
    assert metadata["dataset_index"].tolist() == [0, 1, 2]
    assert metadata[SYMBOL_COLUMN].tolist() == ["BTC/USDT", "BTC/USDT", "ETH/USDT"]


def test_sequence_standardizer_payload_and_transform_are_stable():
    sequences = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )
    standardizer = SequenceStandardizer().fit(sequences)

    transformed = standardizer.transform(sequences)

    np.testing.assert_allclose(standardizer.mean_, [2.5, 25.0])
    np.testing.assert_allclose(standardizer.std_, [1.118034, 11.18034], rtol=1e-6)
    np.testing.assert_allclose(transformed.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(transformed.std(axis=(0, 1)), [1.0, 1.0], rtol=1e-6)
    assert standardizer.to_payload() == {
        "mean": pytest.approx([2.5, 25.0]),
        "std": pytest.approx([1.118034, 11.18034], rel=1e-6),
    }


def test_lstm_train_eval_split_keeps_suffix_for_validation():
    train_indices, eval_indices = split_lstm_train_eval_indices(10, validation_fraction=0.25)

    assert train_indices == list(range(8))
    assert eval_indices == [8, 9]


def test_candle_lstm_train_eval_split_purges_timestamps_before_eval():
    history = build_history_by_symbol(make_history_frame(), ["feature_a", "feature_b"])
    sample_frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range("2025-01-01 01:00:00", periods=6, freq="h"),
            SYMBOL_COLUMN: ["BTC/USDT"] * 6,
            TARGET_COLUMN: [0, 1, 0, 1, 0, 1],
        }
    )
    extended_history = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range("2025-01-01 00:00:00", periods=7, freq="h"),
            SYMBOL_COLUMN: ["BTC/USDT"] * 7,
            "feature_a": np.arange(7, dtype=float),
            "feature_b": np.arange(10, 17, dtype=float),
        }
    )
    dataset = SequenceDataset(
        sample_frame=sample_frame,
        history_by_symbol=build_history_by_symbol(extended_history, ["feature_a", "feature_b"]),
        feature_columns=["feature_a", "feature_b"],
        sequence_length=2,
    )

    train_indices, eval_indices = split_candle_train_eval_indices(
        dataset,
        validation_fraction=1 / 3,
        purge_gap_timestamps=1,
    )

    assert len(dataset) == 6
    assert train_indices == [0, 1, 2]
    assert eval_indices == [4, 5]
    assert [dataset.samples[idx].timestamp for idx in train_indices][-1] < dataset.samples[eval_indices[0]].timestamp


def test_lstm_feature_meta_payload_is_replay_compatible():
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(["2025-01-01", "2025-01-02"]),
            SYMBOL_COLUMN: ["BTC/USDT", "ETH/USDT"],
        }
    )
    args = Namespace(
        model_name="lstm_target",
        sequence_length=12,
        n_splits=5,
        purge_gap=20,
        split_mode="monthly",
        monthly_train_months=6,
        monthly_test_months=1,
        monthly_window_mode="expanding",
    )

    meta = build_lstm_features_meta(
        predictions=predictions,
        feature_columns=["feature_a"],
        symbols=["BTC/USDT", "ETH/USDT"],
        args=args,
    )

    assert meta["feature_columns"] == ["feature_a"]
    assert meta["feature_formulas_artifact"] == "lstm_target_feature_formulas.json"
    assert meta["inverse_label_mapping"] == {"0": -1, "1": 1}
    assert meta["train_period"] == {"start": "2025-01-01 00:00:00", "end": "2025-01-02 00:00:00"}
    assert meta["feature_clip"]["enabled"] is False


def test_candle_lstm_feature_meta_payload_is_replay_compatible():
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(["2025-01-01", "2025-01-03"]),
            SYMBOL_COLUMN: ["BTC/USDT", "ETH/USDT"],
        }
    )
    args = Namespace(
        sequence_length=24,
        n_splits=5,
        purge_gap=20,
        split_mode="monthly",
        monthly_train_months=6,
        monthly_test_months=1,
        monthly_window_mode="expanding",
    )

    meta = build_candle_lstm_features_meta(
        predictions=predictions,
        feature_columns=["log_return_1"],
        symbols=["BTC/USDT", "ETH/USDT"],
        args=args,
    )

    assert meta["feature_columns"] == ["log_return_1"]
    assert meta["wfv_n_splits"] == 5
    assert meta["wfv_split_mode"] == "monthly"
    assert meta["task_type"] == "lstm_candles_binary_directional_walk_forward_oos"
    assert meta["train_period"] == {"start": "2025-01-01 00:00:00", "end": "2025-01-03 00:00:00"}
