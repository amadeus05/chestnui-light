import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config as cfg
import train
from src_refactor.application.pipeline import StoredPredictionSource, predictions_from_frame
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.types import LightGbmInput, ModelSpec, WalkForwardFold
from src_refactor.infrastructure.models.lightgbm.trainer import LightGbmTrainer


def test_feature_clip_bounds_and_application_are_quantile_based(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_FEATURE_CLIP", True)
    monkeypatch.setattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.25)
    monkeypatch.setattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.75)

    frame = pd.DataFrame(
        {
            train.SYMBOL_COLUMN: ["BTC/USDT"] * 5,
            "feature_a": [0.0, 10.0, 20.0, 30.0, 40.0],
            "feature_b": [-100.0, -10.0, 0.0, 10.0, 100.0],
            "feature_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    bounds = train.build_feature_clip_bounds(
        frame,
        [train.SYMBOL_COLUMN, "feature_a", "feature_b", "feature_all_nan"],
    )
    clipped = train.apply_feature_clip_bounds(frame, bounds)

    assert bounds == {
        "feature_a": {"lower": 10.0, "upper": 30.0},
        "feature_b": {"lower": -10.0, "upper": 10.0},
    }
    assert clipped["feature_a"].tolist() == [10.0, 10.0, 20.0, 30.0, 30.0]
    assert clipped["feature_b"].tolist() == [-10.0, -10.0, 0.0, 10.0, 10.0]
    assert frame["feature_a"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_feature_clip_can_be_disabled(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_FEATURE_CLIP", False)
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})

    assert train.build_feature_clip_bounds(frame, ["x"]) == {}
    assert train.apply_feature_clip_bounds(frame, {}) is frame


def test_sample_weights_match_exponential_decay_without_regime_boost(monkeypatch):
    monkeypatch.setattr(cfg, "SAMPLE_WEIGHT_MIN", 0.0)
    monkeypatch.setattr(cfg, "SAMPLE_WEIGHT_MAX", 10.0)
    timestamps = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-11", "2025-01-21"]))

    weights = train.compute_sample_weights(timestamps, half_life_days=10.0, regime_aware=False)

    expected_days_ago = np.array([20.0, 10.0, 0.0])
    expected = np.exp(-(np.log(2.0) / 10.0) * expected_days_ago)
    np.testing.assert_allclose(weights, expected)


def test_sample_weights_apply_recent_and_regime_boosts(monkeypatch):
    monkeypatch.setattr(cfg, "REGIME_AWARE_WEIGHTING", True)
    monkeypatch.setattr(cfg, "REGIME_RECENT_DAYS_BOOST", 7.0)
    monkeypatch.setattr(cfg, "REGIME_RECENT_BOOST_FACTOR", 2.0)
    monkeypatch.setattr(cfg, "REGIME_WEIGHT_STRENGTH", 0.2)
    monkeypatch.setattr(cfg, "REGIME_WEIGHT_STRENGTH_CAP", 0.25)
    monkeypatch.setattr(cfg, "REGIME_WEIGHT_SLOPE_SCALE_4H", 0.08)
    monkeypatch.setattr(cfg, "SAMPLE_WEIGHT_MIN", 0.0)
    monkeypatch.setattr(cfg, "SAMPLE_WEIGHT_MAX", 10.0)
    frame = pd.DataFrame(
        {
            train.TIMESTAMP_COLUMN: pd.to_datetime(["2025-01-01", "2025-01-21"]),
            "market_breadth_ema_fast_slow_1h": [0.5, 1.0],
            "market_breadth_pos_return_4h_3": [0.5, 1.0],
            "ema_slope_4h": [0.0, 1.0],
        }
    )

    weights = train.compute_sample_weights(frame, half_life_days=10.0, regime_aware=True)

    old_base = np.exp(-(np.log(2.0) / 10.0) * 20.0)
    recent_base = 2.0
    recent_regime_multiplier = 1.0 + 0.2
    np.testing.assert_allclose(weights, [old_base, recent_base * recent_regime_multiplier], rtol=1e-6)


def test_internal_eval_plan_selects_time_ordered_suffix(monkeypatch):
    monkeypatch.setattr(cfg, "INTERNAL_EVAL_MIN_FRACTION", 0.25, raising=False)
    monkeypatch.setattr(cfg, "INTERNAL_EVAL_MAX_FRACTION", 0.50, raising=False)
    monkeypatch.setattr(cfg, "INTERNAL_EVAL_STEP_FRACTION", 0.25, raising=False)
    monkeypatch.setattr(cfg, "INTERNAL_EVAL_MAX_CLASS_RATE_DIFF", 0.50, raising=False)
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    plan = train.resolve_internal_eval_plan(labels)

    assert plan["eval_size"] == 2
    assert plan["eval_fraction"] == pytest.approx(0.25)
    assert plan["fit_rate"] == pytest.approx(0.5)
    assert plan["eval_rate"] == pytest.approx(0.5)
    assert plan["was_expanded"] is False


def test_evaluate_model_probability_threshold_metrics_are_stable(monkeypatch):
    monkeypatch.setattr(cfg, "CONFIDENCE_THRESHOLD", 0.60)
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_proba = np.array(
        [
            [0.10, 0.90],
            [0.80, 0.20],
            [0.55, 0.45],
            [0.52, 0.48],
        ]
    )

    metrics = train.evaluate_model(y_true, y_pred, y_proba, split_name="oos")

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["oos_rows"] == 4
    threshold_060 = metrics["probability_threshold_metrics"]["0.60"]
    assert threshold_060["rows"] == 2
    assert threshold_060["coverage"] == pytest.approx(0.5)
    assert threshold_060["long_signals"] == 1
    assert threshold_060["short_signals"] == 1
    assert threshold_060["no_trade"] == 2
    assert threshold_060["signal_confusion_matrix"] == [[1, 0], [0, 1]]


def test_monthly_split_builder_uses_rolling_calendar_windows():
    unique_ts = pd.date_range("2025-01-01", periods=150, freq="D").to_numpy()

    splits = train.build_timestamp_splits(
        unique_ts=unique_ts,
        n_splits=3,
        split_mode="monthly",
        monthly_train_months=2,
        monthly_test_months=1,
        monthly_window_mode="rolling",
    )

    assert [
        (
            fold_idx,
            pd.Timestamp(train_ts[0]),
            pd.Timestamp(train_ts[-1]),
            pd.Timestamp(test_ts[0]),
            pd.Timestamp(test_ts[-1]),
        )
        for fold_idx, train_ts, test_ts in splits
    ] == [
        (1, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
        (2, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")),
        (3, pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30"), pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-30")),
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
