from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import bt
import bt_walk_forward
import config as cfg
import train
from src_refactor.application.pipeline import StoredPredictionSource, predictions_from_frame


def normalize_splits(splits):
    return [
        (
            int(fold_idx),
            pd.to_datetime(train_ts).astype("datetime64[ns]").tolist(),
            pd.to_datetime(test_ts).astype("datetime64[ns]").tolist(),
        )
        for fold_idx, train_ts, test_ts in splits
    ]


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


def test_monthly_split_builders_match_between_train_and_walk_forward():
    unique_ts = pd.date_range("2025-01-01", periods=150, freq="D").to_numpy()

    train_splits = train.build_timestamp_splits(
        unique_ts=unique_ts,
        n_splits=3,
        split_mode="monthly",
        monthly_train_months=2,
        monthly_test_months=1,
        monthly_window_mode="rolling",
    )
    legacy_splits = list(
        bt_walk_forward.iter_monthly_splits(
            unique_ts,
            train_months=2,
            test_months=1,
            window_mode="rolling",
        )
    )

    assert normalize_splits(train_splits) == normalize_splits(legacy_splits)


def test_lightgbm_walk_forward_feature_meta_matches_replay_contract():
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "p_short": [0.2, 0.7],
            "p_long": [0.8, 0.3],
        }
    )
    args = Namespace(
        n_splits=5,
        purge_gap=20,
        split_mode="monthly",
        monthly_train_months=6,
        monthly_test_months=1,
        monthly_window_mode="expanding",
    )
    meta = bt_walk_forward.build_features_meta(
        predictions=predictions,
        feature_columns=["feature_a", "feature_b"],
        symbols=["BTC/USDT", "ETH/USDT"],
        args=args,
    )

    assert meta["feature_columns"] == ["feature_a", "feature_b"]
    assert meta["label_mapping"] == {"short": 0, "long": 1}
    assert meta["inverse_label_mapping"] == {"0": -1, "1": 1}
    assert meta["train_period"] == {"start": "2025-01-01 00:00:00", "end": "2025-01-02 00:00:00"}
    assert meta["wfv_split_mode"] == "monthly"


def test_prediction_lookup_deduplicates_by_last_symbol_timestamp():
    predictions = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "p_short": [0.6, 0.4],
            "p_long": [0.4, 0.6],
        }
    )

    lookup = bt.build_prediction_lookup(predictions)

    assert lookup == {(pd.Timestamp("2025-01-01 00:00:00"), "BTC/USDT"): (0.4, 0.6)}


def test_stored_prediction_source_builds_oos_predictions_from_legacy_frame():
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
    assert converted[0].raw["barrier_stop_pct"] == pytest.approx(0.02)
    assert converted[0].raw["barrier_take_pct"] == pytest.approx(0.04)
    assert pd.Timestamp("2025-01-01 00:00:00") in source.predictions_by_timestamp
