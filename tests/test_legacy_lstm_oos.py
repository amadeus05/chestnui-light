import numpy as np
import pandas as pd

import bt
from lstm.dataset import SequenceDataset
from lstm import train_lstm_walk_forward as lstm_wf
from lstm_candles import bt_lstm_candles_walk_forward as candle_replay
from lstm_candles import data as candle_data


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


def test_backtest_uses_metadata_symbols_for_external_predictions(monkeypatch):
    captured = {}

    def fake_load_all_raw_data(symbols):
        captured["symbols"] = list(symbols)
        return {}

    monkeypatch.setattr(bt, "load_all_raw_data", fake_load_all_raw_data)
    predictions = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01 01:00:00")],
            "symbol": ["ETH/USDT"],
            "p_short": [0.4],
            "p_long": [0.6],
        }
    )
    features_meta = {
        "feature_columns": [],
        "symbols": ["ETH/USDT"],
        "train_period": {
            "start": "2025-01-01 01:00:00",
            "end": "2025-01-01 01:00:00",
        },
        "feature_clip": {"bounds": {}},
    }

    bt.backtest(features_meta=features_meta, predictions=predictions)

    assert captured["symbols"] == ["ETH/USDT"]


def test_candle_lstm_replay_attaches_barriers_for_zero_target_candidates(monkeypatch):
    class FakeRepository:
        def __init__(self, db_path: str):
            self.db_path = db_path

        def load_features(self, symbol):
            return _feature_dataset()

    monkeypatch.setattr(candle_replay, "HistoricalKlineRepository", FakeRepository)
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "symbol": ["BTC/USDT"] * 3,
            "p_short": [0.2, 0.6, 0.4],
            "p_long": [0.8, 0.4, 0.6],
            "fold": [1, 1, 1],
        }
    )

    merged = candle_replay.attach_barrier_columns_from_feature_tables(predictions)

    assert merged["barrier_stop_pct"].tolist() == [0.02, 0.02, 0.02]
    assert merged["barrier_take_pct"].tolist() == [0.04, 0.04, 0.04]


def test_candle_lstm_sequence_frame_uses_decision_timestamps(monkeypatch):
    class FakeRepository:
        def load_candles(self, symbol, timeframe):
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01 00:00:00", periods=30, freq="h"),
                    "open": np.linspace(100, 129, 30),
                    "high": np.linspace(101, 130, 30),
                    "low": np.linspace(99, 128, 30),
                    "close": np.linspace(100.5, 129.5, 30),
                    "volume": np.linspace(1000, 1029, 30),
                }
            )

        def load_funding_rates(self, symbol):
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        def load_premium_index_klines(self, symbol, timeframe):
            return pd.DataFrame(columns=["timestamp", "premium_index_close"])

        def load_open_interest(self, symbol, timeframe):
            return pd.DataFrame(columns=["timestamp", "open_interest"])

    monkeypatch.setattr(candle_data.train, "get_end_date_cutoff", lambda: None)
    repository = FakeRepository()
    btc_context = candle_data._build_btc_context(repository)

    frame = candle_data.build_symbol_sequence_frame(repository, "ETH/USDT", btc_context)

    assert frame["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 01:00:00")
    assert frame["timestamp"].iloc[-1] == pd.Timestamp("2025-01-02 06:00:00")
