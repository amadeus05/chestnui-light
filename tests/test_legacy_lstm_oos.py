import numpy as np
import pandas as pd

import bt
from lstm.dataset import SequenceDataset
from lstm import train_lstm_walk_forward as lstm_wf


def _feature_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "symbol": ["BTC/USDT"] * 6,
            "Target": [-1, 0, 1, 0, -1, 1],
            "feature_a": np.arange(6, dtype=float),
            "feature_b": np.arange(10, 16, dtype=float),
            "barrier_stop_pct": [0.02] * 6,
            "barrier_take_pct": [0.04] * 6,
        }
    )


class _FakeRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_feature_dataset(self, symbols):
        return _feature_dataset()


def test_lstm_frame_loader_keeps_candidates_but_trains_only_directional(monkeypatch):
    monkeypatch.setattr(lstm_wf, "HistoricalKlineRepository", _FakeRepository)
    monkeypatch.setattr(lstm_wf.train, "get_end_date_cutoff", lambda: None)

    candidate_frame, directional_frame, full_frame, history_by_symbol, feature_columns = lstm_wf.load_frames(
        "unused.db",
        ["BTC/USDT"],
    )

    assert len(candidate_frame) == 6
    assert set(candidate_frame["Target"].astype(int)) == {-1, 0, 1}
    assert len(directional_frame) == 4
    assert set(directional_frame["Target"].astype(int)) == {0, 1}
    assert len(full_frame) == 6
    assert feature_columns == ["feature_a", "feature_b"]
    assert set(history_by_symbol) == {"BTC/USDT"}


def test_sequence_dataset_can_predict_candidate_rows_with_zero_targets():
    frame = _feature_dataset()
    feature_columns = ["feature_a", "feature_b"]
    history_by_symbol = lstm_wf.build_history_by_symbol(frame, feature_columns)

    dataset = SequenceDataset(
        frame,
        history_by_symbol,
        feature_columns,
        sequence_length=2,
    )
    metadata = dataset.metadata_frame()

    assert 0 in metadata["Target"].astype(int).tolist()
    assert len(metadata) > int((metadata["Target"].astype(int) != 0).sum())


def test_backtest_prediction_lookup_accepts_candidate_predictions():
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "symbol": ["BTC/USDT"] * 3,
            "p_short": [0.2, 0.6, 0.4],
            "p_long": [0.8, 0.4, 0.6],
            "fold": [1, 1, 1],
            "Target": [-1, 0, 1],
        }
    )

    lookup = bt.build_prediction_lookup(predictions)

    assert len(lookup) == 3
    assert lookup[(pd.Timestamp("2025-01-01 01:00:00"), "BTC/USDT")] == (0.6, 0.4)
