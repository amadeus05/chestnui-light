from __future__ import annotations

from src.models.contracts import ModelKey, ModelSpec


LIGHTGBM_SPEC = ModelSpec(
    key="lightgbm",
    display_name="LightGBM directional classifier",
    artifact_name="lightgbm_target",
    model_type="lightgbm",
    feature_source="etl_features",
    supports_production=True,
    supports_walk_forward=True,
    supports_backtest_replay=True,
    default_predictions_name="walk_forward_oos_predictions",
    default_chart_name="equity_curve_walk_forward.png",
    legacy_train_module="train",
    legacy_wfv_module="bt_walk_forward",
    legacy_backtest_module="bt",
)

LSTM_SPEC = ModelSpec(
    key="lstm",
    display_name="LSTM classifier on ETL feature sequences",
    artifact_name="lstm_target",
    model_type="lstm",
    feature_source="etl_sequences",
    supports_production=True,
    supports_walk_forward=True,
    supports_backtest_replay=True,
    default_predictions_name="lstm_walk_forward_oos_predictions",
    default_chart_name="equity_curve_lstm_walk_forward.png",
    legacy_wfv_module="lstm.train_lstm_walk_forward",
    legacy_backtest_module="lstm.bt_lstm_walk_forward",
    legacy_production_module="lstm.train_lstm_production",
)

LSTM_CANDLES_SPEC = ModelSpec(
    key="lstm_candles",
    display_name="LSTM classifier on candle sequences",
    artifact_name="lstm_candles_target",
    model_type="lstm_candles",
    feature_source="candle_sequences",
    supports_production=True,
    supports_walk_forward=True,
    supports_backtest_replay=True,
    default_predictions_name="lstm_candles_walk_forward_oos_predictions",
    default_chart_name="equity_curve_lstm_candles_walk_forward.png",
    legacy_wfv_module="lstm_candles.train_lstm_candles_walk_forward",
    legacy_backtest_module="lstm_candles.bt_lstm_candles_walk_forward",
)

MODEL_REGISTRY: dict[ModelKey, ModelSpec] = {
    LIGHTGBM_SPEC.key: LIGHTGBM_SPEC,
    LSTM_SPEC.key: LSTM_SPEC,
    LSTM_CANDLES_SPEC.key: LSTM_CANDLES_SPEC,
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]  # type: ignore[index]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{key}'. Available models: {available}") from exc


def list_model_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())
