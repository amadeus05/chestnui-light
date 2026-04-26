import numpy as np
import pytest

from src.ml.training.artifacts import build_payload_fingerprint
from src.ml.training.diagnostics import build_fold_stability_payload
from src.ml.training.metrics import evaluate_model
from src.ml.training.walk_forward import resolve_effective_purge_gap


def test_effective_purge_gap_covers_adaptive_horizon(monkeypatch):
    monkeypatch.setattr("config.HORIZON", 12)
    monkeypatch.setattr("config.ENABLE_ADAPTIVE_HORIZON", True)
    monkeypatch.setattr("config.ADAPTIVE_HORIZON_MAX", 20)

    effective_purge_gap, label_horizon = resolve_effective_purge_gap(12)

    assert label_horizon == 20
    assert effective_purge_gap == 20


def test_evaluate_model_returns_nullable_auc_for_one_class():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_proba = np.array(
        [
            [0.10, 0.90],
            [0.20, 0.80],
            [0.15, 0.85],
        ]
    )

    metrics = evaluate_model(y_true, y_pred, y_proba, split_name="test")

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["mcc"] is None
    assert metrics["test_rows"] == 3


def test_fold_stability_ignores_missing_fold_auc():
    stability = build_fold_stability_payload(
        [
            {"accuracy": 0.50, "roc_auc": None},
            {"accuracy": 0.75, "roc_auc": 0.60},
            {"accuracy": 1.00, "roc_auc": 0.70},
        ]
    )

    assert stability["accuracy_range"] == 0.5
    assert stability["roc_auc_range"] == pytest.approx(0.1)


def test_feature_payload_fingerprint_ignores_generated_at():
    payload = {
        "model_name": "model",
        "generated_at_utc": "2026-01-01 00:00:00 UTC",
        "features": [{"name": "ema_fast_slow"}],
    }
    updated_timestamp_payload = {
        **payload,
        "generated_at_utc": "2026-01-02 00:00:00 UTC",
    }

    assert build_payload_fingerprint(payload) == build_payload_fingerprint(updated_timestamp_payload)
