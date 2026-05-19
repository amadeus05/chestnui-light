import numpy as np
import pandas as pd
import pytest

from src_refactor.application.pipeline.prediction_source import ModelPredictionSource
from src_refactor.application.pipeline.runtime_market_cache import RuntimeMarketCache
from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelInputBuilder, ModelInputRequest, ModelPredictor
from src_refactor.core.types import LightGbmInput, ModelInput, ModelSpec, Prediction
from src_refactor.infrastructure.models.lightgbm.input_builder import LightGbmInputBuilder
from src_refactor.infrastructure.models.lstm_features.input_builder import LstmFeatureInputBuilder


def make_candles(rows: int = 72, *, symbol: str = "BTC/USDT", freq: str = "h", offset: float = 0.0) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq=freq)
    close = 100.0 + offset + np.arange(rows, dtype=float) * 0.25
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "timeframe": "4h" if freq == "4h" else "1h",
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


class CapturingInputBuilder(ModelInputBuilder):
    def __init__(self) -> None:
        self.request: ModelInputRequest | None = None

    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> ModelInput:
        raise NotImplementedError

    def build_predict_input(self, request: ModelInputRequest | pd.DataFrame, spec: ModelSpec | None = None) -> ModelInput:
        assert isinstance(request, ModelInputRequest)
        self.request = request
        return LightGbmInput(features=pd.DataFrame({"x": [1.0]}), feature_names=("x",))


class FrameInputBuilder(ModelInputBuilder):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> ModelInput:
        raise NotImplementedError

    def build_predict_input(self, request: ModelInputRequest | pd.DataFrame, spec: ModelSpec | None = None) -> ModelInput:
        assert isinstance(request, ModelInputRequest)
        return LightGbmInput(
            features=pd.DataFrame({"x": [1.0]}),
            feature_names=("x",),
            metadata={"frame": self.frame.copy()},
        )


class StaticPredictor(ModelPredictor):
    def predict(self, model_input: ModelInput) -> Prediction:
        return Prediction(
            timestamp=pd.Timestamp("2025-01-01"),
            symbol="",
            timeframe="1h",
            model_id="raw",
            direction=0,
            confidence=0.8,
            proba_long=0.8,
            proba_short=0.2,
        )


def test_model_prediction_source_uses_model_input_request():
    cache = RuntimeMarketCache()
    contexts = cache.update_from_frame(make_candles(rows=3))
    builder = CapturingInputBuilder()
    source = ModelPredictionSource(
        input_builder=builder,
        predictor=StaticPredictor(),
        model_spec=ModelSpec(model_type="lightgbm", timeframe="1h"),
    )

    predictions = source.predictions_for(contexts[-1])

    assert len(predictions) == 1
    assert builder.request is not None
    assert builder.request.symbol == "BTC/USDT"
    assert builder.request.timeframe == "1h"
    assert builder.request.base_candles["close"].tolist() == [100.0, 100.25]


def test_model_prediction_source_attaches_barriers_from_model_frame():
    cache = RuntimeMarketCache()
    contexts = cache.update_from_frame(make_candles(rows=3))
    frame = make_candles(rows=3)
    frame["barrier_stop_pct"] = [0.01, 0.02, 0.03]
    frame["barrier_take_pct"] = [0.02, 0.04, 0.06]
    source = ModelPredictionSource(
        input_builder=FrameInputBuilder(frame),
        predictor=StaticPredictor(),
        model_spec=ModelSpec(model_type="lightgbm", timeframe="1h"),
    )

    prediction = source.predictions_for(contexts[-1])[0]

    assert prediction.signal_gap == pytest.approx(0.6)
    assert prediction.stop_pct == pytest.approx(0.03)
    assert prediction.take_pct == pytest.approx(0.06)


def test_model_prediction_source_attaches_runtime_barriers_from_labeling_metadata():
    cache = RuntimeMarketCache()
    contexts = cache.update_from_frame(make_candles(rows=3))
    source = ModelPredictionSource(
        input_builder=FrameInputBuilder(make_candles(rows=3)),
        predictor=StaticPredictor(),
        model_spec=ModelSpec(
            model_type="lightgbm",
            timeframe="1h",
            metadata={
                "labeling": {
                    "use_dynamic_barriers": False,
                    "sl_pct": 0.025,
                    "tp_pct": 0.05,
                }
            },
        ),
    )

    prediction = source.predictions_for(contexts[-1])[0]

    assert prediction.stop_pct == pytest.approx(0.025)
    assert prediction.take_pct == pytest.approx(0.05)


def test_model_prediction_source_passes_causal_cross_symbol_maps():
    frame = pd.concat(
        [
            make_candles(rows=2, symbol="BTC/USDT", offset=0.0),
            make_candles(rows=2, symbol="ETH/USDT", offset=50.0),
        ],
        ignore_index=True,
    )
    cache = RuntimeMarketCache()
    contexts = cache.update_from_frame(frame)
    builder = CapturingInputBuilder()
    source = ModelPredictionSource(
        input_builder=builder,
        predictor=StaticPredictor(),
        model_spec=ModelSpec(model_type="lightgbm", timeframe="1h"),
        htf_candle_map={"ETH/USDT": make_candles(rows=8, symbol="ETH/USDT", freq="4h")},
    )

    source.predictions_for(contexts[0])

    assert builder.request is not None
    assert set(builder.request.base_candle_map) == {"BTC/USDT", "ETH/USDT"}
    assert builder.request.base_candle_map["BTC/USDT"]["close"].tolist() == [100.0]
    assert builder.request.base_candle_map["ETH/USDT"]["close"].tolist() == [150.0]
    assert "ETH/USDT" in builder.request.htf_candle_map


def test_lightgbm_builder_builds_runtime_features_from_raw_candles():
    spec = ModelSpec(
        model_type="lightgbm",
        timeframe="1h",
        metadata={"feature_columns": ["ema_fast_slow"]},
    )

    model_input = LightGbmInputBuilder().build_predict_input(
        ModelInputRequest(
            symbol="BTC/USDT",
            timeframe="1h",
            base_candles=make_candles(),
            spec=spec,
        )
    )

    assert model_input.feature_names == ("ema_fast_slow",)
    assert list(model_input.features.columns) == ["ema_fast_slow"]
    assert len(model_input.features) == 72


def test_lstm_feature_builder_builds_runtime_feature_window_from_raw_candles():
    spec = ModelSpec(
        model_type="lstm_features",
        timeframe="1h",
        metadata={"feature_columns": ["ema_fast_slow"], "window_size": 8},
    )

    model_input = LstmFeatureInputBuilder().build_predict_input(
        ModelInputRequest(
            symbol="BTC/USDT",
            timeframe="1h",
            base_candles=make_candles(),
            spec=spec,
        )
    )

    assert model_input.feature_names == ("ema_fast_slow",)
    assert model_input.sequence.shape == (1, 8, 1)


def test_lightgbm_builder_builds_btc_relative_features_from_runtime_symbol_map():
    spec = ModelSpec(
        model_type="lightgbm",
        timeframe="1h",
        metadata={"feature_columns": ["relative_strength_vs_btc_24h"]},
    )
    btc = make_candles(rows=72, symbol="BTC/USDT", offset=0.0)
    eth = make_candles(rows=72, symbol="ETH/USDT", offset=50.0)

    model_input = LightGbmInputBuilder().build_predict_input(
        ModelInputRequest(
            symbol="ETH/USDT",
            timeframe="1h",
            base_candles=eth,
            spec=spec,
            base_candle_map={"BTC/USDT": btc, "ETH/USDT": eth},
        )
    )

    assert model_input.feature_names == ("relative_strength_vs_btc_24h",)
    assert list(model_input.features.columns) == ["relative_strength_vs_btc_24h"]
    assert model_input.features["relative_strength_vs_btc_24h"].notna().any()


def test_lightgbm_builder_merges_htf_features_from_runtime_htf_map():
    spec = ModelSpec(
        model_type="lightgbm",
        timeframe="1h",
        metadata={"feature_columns": ["ema_slope_4h"]},
    )
    base = make_candles(rows=120, symbol="ETH/USDT")
    htf = make_candles(rows=40, symbol="ETH/USDT", freq="4h")

    model_input = LightGbmInputBuilder().build_predict_input(
        ModelInputRequest(
            symbol="ETH/USDT",
            timeframe="1h",
            base_candles=base,
            spec=spec,
            htf_candles=htf,
            htf_candle_map={"ETH/USDT": htf},
        )
    )

    assert model_input.feature_names == ("ema_slope_4h",)
    assert list(model_input.features.columns) == ["ema_slope_4h"]
