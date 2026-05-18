from __future__ import annotations

from src_refactor.infrastructure.models.lightgbm import create_lightgbm_bundle
from src_refactor.infrastructure.models.lstm_candles import create_lstm_candles_bundle
from src_refactor.infrastructure.models.lstm_features import create_lstm_features_bundle
from src_refactor.infrastructure.models.registry import ModelRegistry


def create_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("lightgbm", create_lightgbm_bundle)
    registry.register("lstm_features", create_lstm_features_bundle)
    registry.register("lstm_candles", create_lstm_candles_bundle)
    return registry
