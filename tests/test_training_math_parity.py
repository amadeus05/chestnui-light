import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bt
import etl
import train
from src_refactor.application.pipeline import StoredPredictionSource, predictions_from_frame
from src_refactor.application.training.walk_forward_splitter import WalkForwardSplitter
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import LightGbmInput, ModelSpec, WalkForwardFold
from src_refactor.infrastructure.models.lightgbm.config import LightGbmTrainingConfig
from src_refactor.infrastructure.models.lightgbm.input_builder import LightGbmInputBuilder, SYMBOL_COLUMN
from src_refactor.infrastructure.models.lightgbm.trainer import LightGbmTrainer
from src_refactor.infrastructure.models.lightgbm.trainer import compute_sample_weights


def test_feature_clip_bounds_and_application_are_quantile_based():
    frame = pd.DataFrame(
        {
            SYMBOL_COLUMN: ["BTC/USDT"] * 5,
            "feature_a": [0.0, 10.0, 20.0, 30.0, 40.0],
            "feature_b": [-100.0, -10.0, 0.0, 10.0, 100.0],
            "feature_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    config = LightGbmTrainingConfig(
        enable_feature_clip=True,
        feature_clip_lower_q=0.25,
        feature_clip_upper_q=0.75,
    )

    bounds = LightGbmInputBuilder.build_feature_clip_bounds(
        frame,
        [SYMBOL_COLUMN, "feature_a", "feature_b", "feature_all_nan"],
        config,
    )
    clipped = LightGbmInputBuilder.apply_feature_clip_bounds(frame, bounds)

    assert bounds == {
        "feature_a": {"lower": 10.0, "upper": 30.0},
        "feature_b": {"lower": -10.0, "upper": 10.0},
    }
    assert clipped["feature_a"].tolist() == [10.0, 10.0, 20.0, 30.0, 30.0]
    assert clipped["feature_b"].tolist() == [-10.0, -10.0, 0.0, 10.0, 10.0]
    assert frame["feature_a"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_feature_clip_can_be_disabled():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    config = LightGbmTrainingConfig(enable_feature_clip=False)

    assert LightGbmInputBuilder.build_feature_clip_bounds(frame, ["x"], config) == {}
    assert LightGbmInputBuilder.apply_feature_clip_bounds(frame, {}) is frame


def test_legacy_evaluate_model_handles_one_class_labels():
    metrics = train.evaluate_model(
        y_true=np.array([1, 1, 1]),
        y_pred=np.array([1, 1, 0]),
        y_proba=np.array(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        ),
        split_name="oos",
    )

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["oos_rows"] == 3


def test_sample_weights_match_exponential_decay_without_regime_boost():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01", "2025-01-11", "2025-01-21"])})
    config = LightGbmTrainingConfig.from_metadata(
        {
            "sample_weight_half_life_days": 10.0,
            "regime_aware_weighting": False,
            "sample_weight_min": 0.0,
            "sample_weight_max": 10.0,
        }
    )

    weights = compute_sample_weights(frame, config)

    expected_days_ago = np.array([20.0, 10.0, 0.0])
    expected = np.exp(-(np.log(2.0) / 10.0) * expected_days_ago)
    np.testing.assert_allclose(weights, expected)


def test_sample_weights_apply_recent_and_regime_boosts():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-21"]),
            "market_breadth_ema_fast_slow_1h": [0.5, 1.0],
            "market_breadth_pos_return_4h_3": [0.5, 1.0],
            "ema_slope_4h": [0.0, 1.0],
        }
    )
    config = LightGbmTrainingConfig.from_metadata(
        {
            "sample_weight_half_life_days": 10.0,
            "regime_aware_weighting": True,
            "regime_recent_days_boost": 7.0,
            "regime_recent_boost_factor": 2.0,
            "regime_weight_strength": 0.2,
            "regime_weight_strength_cap": 0.25,
            "regime_weight_slope_scale_4h": 0.08,
            "sample_weight_min": 0.0,
            "sample_weight_max": 10.0,
        }
    )

    weights = compute_sample_weights(frame, config)

    old_base = np.exp(-(np.log(2.0) / 10.0) * 20.0)
    recent_base = 2.0
    recent_regime_multiplier = 1.0 + 0.2
    np.testing.assert_allclose(weights, [old_base, recent_base * recent_regime_multiplier], rtol=1e-6)


def test_etl_feature_timestamps_are_close_time_decision_timestamps():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }
    )

    shifted = etl.with_decision_timestamps(frame, "1h")

    assert shifted["open_time"].tolist() == frame["timestamp"].tolist()
    assert shifted["timestamp"].tolist() == pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 02:00:00"]).tolist()


def test_backtest_maps_decision_timestamp_to_execution_open_time():
    decision_ts = pd.Timestamp("2025-01-01 01:00:00")

    assert bt.execution_timestamp_for_decision_time(decision_ts, "1h") == pd.Timestamp("2025-01-01 00:00:00")


def test_backtest_execution_window_maps_decision_period_to_open_time_grid():
    start, end = bt.build_execution_window(
        pd.Timestamp("2025-01-01 01:00:00"),
        pd.Timestamp("2025-01-01 03:00:00"),
        "1h",
    )

    assert start == pd.Timestamp("2025-01-01 00:00:00")
    assert end == pd.Timestamp("2025-01-01 02:00:00")


def test_legacy_finalize_feature_frame_reports_dropna_row_loss(monkeypatch, caplog):
    monkeypatch.setattr(etl.cfg, "HORIZON", 0)
    monkeypatch.setattr(etl.cfg, "ENABLE_ADAPTIVE_HORIZON", False, raising=False)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
            "feature_a": [1.0, np.nan, 3.0, 4.0],
            "feature_b": [np.inf, 2.0, 3.0, 4.0],
            "barrier_stop_pct": [0.02, 0.02, 0.02, 0.02],
            "barrier_take_pct": [0.04, 0.04, 0.04, 0.04],
            "Target": [1, -1, 1, -1],
        }
    )

    with caplog.at_level("WARNING", logger=etl.logger.name):
        finalized = etl.finalize_feature_frame(frame, ["feature_a", "feature_b"])

    assert len(finalized) == 1
    assert "Feature finalize dropna removed rows" in caplog.text
    assert "dropna_rows=2" in caplog.text
    assert "feature_a" in caplog.text
    assert "feature_b" in caplog.text


def test_legacy_internal_eval_split_applies_purge_gap(monkeypatch):
    monkeypatch.setattr(train.cfg, "effective_max_label_horizon", lambda: 3, raising=False)

    slices = train.resolve_internal_eval_slices(n_rows=20, eval_size=4)

    assert slices["eval_start"] == 16
    assert slices["fit_end"] == 13
    assert slices["requested_purge_gap"] == 3
    assert slices["applied_purge_gap"] == 3
    assert slices["purge_reduced"] is False


def test_legacy_internal_eval_split_reduces_purge_when_train_is_too_small(monkeypatch):
    monkeypatch.setattr(train.cfg, "effective_max_label_horizon", lambda: 10, raising=False)

    slices = train.resolve_internal_eval_slices(n_rows=8, eval_size=4)

    assert slices["eval_start"] == 4
    assert slices["fit_end"] == 1
    assert slices["requested_purge_gap"] == 10
    assert slices["applied_purge_gap"] == 3
    assert slices["purge_reduced"] is True


def test_monthly_split_builder_uses_rolling_calendar_windows():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=150, freq="D")})
    config = ExperimentConfig(
        model=ModelSpec(model_type="lightgbm", timeframe="1d"),
        split_mode="monthly_rolling",
        n_splits=3,
        train_months=2,
        test_months=1,
    )
    splits = WalkForwardSplitter(config).split(frame)

    assert [
        (
            fold.fold_id,
            fold.train_start,
            fold.train_end,
            fold.test_start,
            fold.test_end,
        )
        for fold in splits
    ] == [
        (0, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
        (1, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")),
        (2, pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30"), pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-30")),
    ]


def test_lightgbm_trainer_metadata_keeps_replay_contract(monkeypatch):
    class FakeLightGbmClassifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, x_train, y_train, sample_weight=None, categorical_feature=None):
            self.x_train = x_train
            self.y_train = y_train
            self.sample_weight = sample_weight
            self.categorical_feature = categorical_feature
            return self

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(LGBMClassifier=FakeLightGbmClassifier))

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "Target": [1, -1],
        }
    )
    train_input = LightGbmInput(
        features=frame[["feature_a", "feature_b"]],
        feature_names=("feature_a", "feature_b"),
        metadata={"frame": frame, "clip_bounds": {"feature_a": {"lower": 0.0, "upper": 10.0}}},
    )
    config = ExperimentConfig(
        model=ModelSpec(model_type="lightgbm", timeframe="1h", profile="baseline", metadata={"labeling": {"horizon": 16}}),
        symbols=("BTC/USDT", "ETH/USDT"),
        n_splits=5,
        purge_gap=20,
        split_mode="monthly_expanding",
        train_months=6,
        test_months=1,
    )
    fold = WalkForwardFold(
        fold_id=2,
        train_start=pd.Timestamp("2025-01-01"),
        train_end=pd.Timestamp("2025-01-02"),
        test_start=pd.Timestamp("2025-02-01"),
        test_end=pd.Timestamp("2025-02-28"),
    )
    artifact = LightGbmTrainer().train(train_input, config, fold)

    assert artifact.fold_id == 2
    assert artifact.metadata["feature_columns"] == ["feature_a", "feature_b"]
    assert artifact.metadata["feature_clip"] == {"bounds": {"feature_a": {"lower": 0.0, "upper": 10.0}}}
    assert artifact.metadata["label_mapping"] == {"short": 0, "long": 1}
    assert artifact.metadata["inverse_label_mapping"] == {"0": -1, "1": 1}
    assert artifact.metadata["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert artifact.metadata["labeling"] == {"horizon": 16}


def test_prediction_lookup_deduplicates_by_last_symbol_timestamp():
    predictions = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "p_short": [0.6, 0.4],
            "p_long": [0.4, 0.6],
            "barrier_stop_pct": [0.02, 0.02],
            "barrier_take_pct": [0.04, 0.04],
        }
    )

    source = StoredPredictionSource.from_frame(
        predictions,
        model_id="walk_forward_oos",
        timeframe="1h",
    )
    timestamp = pd.Timestamp("2025-01-01 00:00:00")
    deduped = source.predictions_by_timestamp[timestamp][0]

    assert len(source.predictions_by_timestamp[timestamp]) == 1
    assert deduped.proba_short == pytest.approx(0.4)
    assert deduped.proba_long == pytest.approx(0.6)


def test_backtest_rejects_non_positive_barriers():
    zero_stop = pd.DataFrame({"barrier_stop_pct": [0.0], "barrier_take_pct": [0.04]})
    negative_take = pd.DataFrame({"barrier_stop_pct": [0.02], "barrier_take_pct": [-0.04]})

    assert bt.get_barrier_pcts(zero_stop) == (None, None)
    assert bt.get_barrier_pcts(negative_take) == (None, None)


def test_stored_prediction_source_builds_oos_predictions_from_prediction_frame():
    predictions = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "p_short": [0.6, 0.4],
            "p_long": [0.4, 0.6],
        }
    )
    barriers = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00"],
            "symbol": ["BTC/USDT"],
            "barrier_stop_pct": [0.02],
            "barrier_take_pct": [0.04],
        }
    )

    converted = predictions_from_frame(
        predictions,
        model_id="walk_forward_oos",
        timeframe="1h",
        barrier_frame=barriers,
    )
    source = StoredPredictionSource.from_frame(
        predictions,
        model_id="walk_forward_oos",
        timeframe="1h",
        barrier_frame=barriers,
    )

    assert len(converted) == 1
    assert converted[0].proba_short == pytest.approx(0.4)
    assert converted[0].proba_long == pytest.approx(0.6)
    assert converted[0].signal_gap == pytest.approx(0.2)
    assert converted[0].stop_pct == pytest.approx(0.02)
    assert converted[0].take_pct == pytest.approx(0.04)
    assert pd.Timestamp("2025-01-01 00:00:00") in source.predictions_by_timestamp
