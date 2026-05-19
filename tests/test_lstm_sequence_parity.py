import numpy as np
import pandas as pd
import pytest

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import LstmCandleWindowInput, LstmFeatureSequenceInput, ModelSpec, WalkForwardFold
from src_refactor.infrastructure.models.lstm_candles.input_builder import LstmCandleInputBuilder
from src_refactor.infrastructure.models.lstm_candles import trainer as candle_trainer_module
from src_refactor.infrastructure.models.lstm_candles.trainer import LstmCandleTrainer
from src_refactor.infrastructure.models.lstm_common import SequenceStandardizer, split_train_eval_indices
from src_refactor.infrastructure.models.lstm_features.input_builder import LstmFeatureInputBuilder
from src_refactor.infrastructure.models.lstm_features import trainer as feature_trainer_module
from src_refactor.infrastructure.models.lstm_features.trainer import LstmFeatureTrainer

TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
TARGET_COLUMN = "Target"


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2025-01-01 02:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                ]
            ),
            SYMBOL_COLUMN: ["BTC/USDT", "BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT", "ETH/USDT"],
            TARGET_COLUMN: [-1, -1, 1, -1, 1, -1],
            "feature_a": [3.0, 1.0, 20.0, 10.0, 11.0, 12.0],
            "feature_b": [30.0, np.nan, 200.0, 100.0, np.inf, 120.0],
        }
    )


def make_candle_frame(rows: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range("2025-01-01", periods=rows, freq="h"),
            SYMBOL_COLUMN: ["BTC/USDT"] * rows,
            TARGET_COLUMN: [-1, 1, -1, 1][:rows],
        }
    )
    for index, column in enumerate(LstmCandleInputBuilder().candle_columns):
        frame[column] = np.arange(rows, dtype=float) + float(index)
    return frame


def test_lstm_feature_input_builder_uses_sorted_current_and_past_rows_only():
    config = ExperimentConfig(
        model=ModelSpec(model_type="lstm_features", timeframe="1h", metadata={"window_size": 2})
    )

    model_input = LstmFeatureInputBuilder(window_size=2).build_train_input(make_feature_frame(), config)

    assert model_input.feature_names == ("feature_a", "feature_b")
    assert model_input.sequence.shape == (4, 2, 2)
    np.testing.assert_allclose(
        model_input.sequence[0],
        np.array([[1.0, 0.0], [20.0, 200.0]], dtype=np.float32),
    )
    assert model_input.targets.tolist() == [1, 0, 1, 0]


def test_lstm_candle_input_builder_uses_configured_candle_columns():
    config = ExperimentConfig(
        model=ModelSpec(model_type="lstm_candles", timeframe="1h", metadata={"window_size": 2})
    )

    model_input = LstmCandleInputBuilder(window_size=2).build_train_input(make_candle_frame(), config)

    assert model_input.candle_columns == LstmCandleInputBuilder().candle_columns
    assert model_input.candles.shape == (3, 2, len(model_input.candle_columns))
    assert model_input.targets.tolist() == [1, 0, 1]


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
    train_indices, eval_indices = split_train_eval_indices(10, validation_fraction=0.25)

    assert train_indices == list(range(8))
    assert eval_indices == [8, 9]


def test_lstm_feature_trainer_metadata_is_replay_compatible(monkeypatch):
    monkeypatch.setattr(feature_trainer_module, "train_lstm_classifier", _fake_train_lstm_classifier)
    train_input = LstmFeatureSequenceInput(
        sequence=np.ones((2, 3, 2), dtype=np.float32),
        feature_names=("feature_a", "feature_b"),
        window_size=3,
        targets=np.array([0, 1], dtype=np.int64),
    )
    config = ExperimentConfig(
        model=ModelSpec(
            model_type="lstm_features",
            timeframe="1h",
            metadata={"min_train_rows": 1, "window_size": 3, "labeling": {"horizon": 16}},
        ),
        symbols=("BTC/USDT", "ETH/USDT"),
    )

    artifact = LstmFeatureTrainer().train(train_input, config, _fold())

    assert artifact.metadata["feature_columns"] == ["feature_a", "feature_b"]
    assert artifact.metadata["inverse_label_mapping"] == {"0": -1, "1": 1}
    assert artifact.metadata["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert artifact.metadata["model_args"]["input_size"] == 2
    assert artifact.metadata["labeling"] == {"horizon": 16}


def test_lstm_candle_trainer_metadata_is_replay_compatible(monkeypatch):
    monkeypatch.setattr(candle_trainer_module, "train_lstm_classifier", _fake_train_lstm_classifier)
    train_input = LstmCandleWindowInput(
        candles=np.ones((2, 3, 2), dtype=np.float32),
        candle_columns=("log_return_1", "volume_norm_24"),
        window_size=3,
        targets=np.array([0, 1], dtype=np.int64),
    )
    config = ExperimentConfig(
        model=ModelSpec(
            model_type="lstm_candles",
            timeframe="1h",
            metadata={"min_train_rows": 1, "window_size": 3, "labeling": {"horizon": 16}},
        ),
        symbols=("BTC/USDT", "ETH/USDT"),
    )

    artifact = LstmCandleTrainer().train(train_input, config, _fold())

    assert artifact.metadata["feature_columns"] == ["log_return_1", "volume_norm_24"]
    assert artifact.metadata["inverse_label_mapping"] == {"0": -1, "1": 1}
    assert artifact.metadata["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert artifact.metadata["model_args"]["input_size"] == 2
    assert artifact.metadata["labeling"] == {"horizon": 16}


def _fold() -> WalkForwardFold:
    return WalkForwardFold(
        fold_id=0,
        train_start=pd.Timestamp("2025-01-01"),
        train_end=pd.Timestamp("2025-01-02"),
        test_start=pd.Timestamp("2025-01-03"),
        test_end=pd.Timestamp("2025-01-04"),
    )


def _fake_train_lstm_classifier(sequences, targets, config, *, device):
    return _FakeModel(), SequenceStandardizer().fit(sequences), {"best_epoch": 1, "best_eval_loss": 0.25}


class _FakeModel:
    def state_dict(self):
        return {}
