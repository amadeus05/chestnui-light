import numpy as np
import pandas as pd

from src_refactor.domain.features import FeaturePipeline, FeaturePipelineConfig
from src_refactor.domain.labels import LabelingConfig, attach_barrier_columns

REGIME_FEATURES = ["realized_vol_1h", "volatility_regime_stability"]

CAUSAL_FEATURES = [
    "ema_fast_slow",
    "trend_efficiency_24h",
    "realized_vol_1h",
    "volatility_regime_change_1h",
    "volume_ratio_1h",
    "volatility_regime_stability",
    "vol_regime_classification",
    "funding_rate_8h",
    "premium_index_1h",
    "open_interest_change_pct_8h",
    "realized_vol_4h_returns_20",
    "ema_slope_4h",
    "breakout_quality_4h",
    "market_breadth_ema_fast_slow_1h",
    "trend_alignment_1h_4h",
]


def feature_pipeline(feature_columns: list[str]) -> FeaturePipeline:
    return FeaturePipeline(
        FeaturePipelineConfig(
            raw_request={
                "profile": "empty",
                "include_features": feature_columns,
            }
        )
    )


def make_candles(rows: int = 260, freq: str = "h", symbol_offset: float = 0.0) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq=freq)
    idx = np.arange(rows, dtype=float)
    returns = 0.001 + 0.002 * np.sin(idx / 9.0 + symbol_offset)
    close = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1_000.0 + 50.0 * np.cos(idx / 7.0 + symbol_offset),
        }
    )
    frame["funding_rate"] = 0.0001 * np.sin(idx / 13.0 + symbol_offset)
    frame["premium_index_close"] = 1.0 + 0.0003 * np.cos(idx / 11.0 + symbol_offset)
    frame["open_interest"] = 10_000.0 + np.cumsum(3.0 + np.sin(idx / 5.0 + symbol_offset))
    return frame


def perturb_future(frame: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
    output = frame.copy()
    future_mask = output.index > cutoff_idx
    future_multiplier = np.linspace(1.15, 2.25, int(future_mask.sum()))
    output.loc[future_mask, "close"] = output.loc[future_mask, "close"].to_numpy() * future_multiplier
    output.loc[future_mask, "open"] = output.loc[future_mask, "close"] * 0.999
    output.loc[future_mask, "high"] = output.loc[future_mask, "close"] * 1.004
    output.loc[future_mask, "low"] = output.loc[future_mask, "close"] * 0.996
    output.loc[future_mask, "volume"] = output.loc[future_mask, "volume"].to_numpy() * 3.0
    if "funding_rate" in output.columns:
        output.loc[future_mask, "funding_rate"] = output.loc[future_mask, "funding_rate"].to_numpy() * -5.0
    if "premium_index_close" in output.columns:
        output.loc[future_mask, "premium_index_close"] = output.loc[future_mask, "premium_index_close"].to_numpy() * 1.05
    if "open_interest" in output.columns:
        output.loc[future_mask, "open_interest"] = output.loc[future_mask, "open_interest"].to_numpy() * 2.0
    return output


def build_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    return feature_pipeline(REGIME_FEATURES).build_for_symbol("TEST/USDT", frame)


def test_realized_vol_and_regime_stability_do_not_change_when_future_changes():
    cutoff_idx = 130
    base = make_candles()
    altered = perturb_future(base, cutoff_idx)

    base_features = build_regime_features(base)
    altered_features = build_regime_features(altered)

    columns = ["realized_vol_1h", "volatility_regime_stability"]
    pd.testing.assert_frame_equal(
        base_features.loc[:cutoff_idx, columns],
        altered_features.loc[:cutoff_idx, columns],
        check_dtype=False,
    )


def test_dynamic_barriers_do_not_change_when_future_changes():
    labeling_config = LabelingConfig(
        horizon=16,
        use_dynamic_barriers=True,
        barrier_min_pct=0.01,
        barrier_max_pct=0.05,
    )

    cutoff_idx = 130
    base = make_candles()
    altered = perturb_future(base, cutoff_idx)

    base_with_features = build_regime_features(base)
    altered_with_features = build_regime_features(altered)

    base_barriers = attach_barrier_columns(base_with_features, labeling_config)
    altered_barriers = attach_barrier_columns(altered_with_features, labeling_config)

    columns = ["barrier_stop_pct", "barrier_take_pct"]
    pd.testing.assert_frame_equal(
        base_barriers.loc[:cutoff_idx, columns],
        altered_barriers.loc[:cutoff_idx, columns],
        check_dtype=False,
    )


def make_symbol_maps():
    symbols = ["BTC/USDT", "DOGE/USDT", "ETH/USDT"]
    base_map = {
        symbol: make_candles(rows=260, freq="h", symbol_offset=idx * 0.7)
        for idx, symbol in enumerate(symbols)
    }
    htf_map = {
        symbol: make_candles(rows=80, freq="4h", symbol_offset=idx * 0.7)
        for idx, symbol in enumerate(symbols)
    }
    return base_map, htf_map


def perturb_symbol_maps(base_map, htf_map, base_cutoff_idx: int, htf_cutoff_idx: int):
    altered_base = {
        symbol: perturb_future(frame, base_cutoff_idx)
        for symbol, frame in base_map.items()
    }
    altered_htf = {
        symbol: perturb_future(frame, htf_cutoff_idx)
        for symbol, frame in htf_map.items()
    }
    return altered_base, altered_htf


def test_feature_pipeline_is_causal_for_requested_features():
    base_cutoff_idx = 180
    htf_cutoff_idx = 45
    base_map, htf_map = make_symbol_maps()
    altered_base_map, altered_htf_map = perturb_symbol_maps(base_map, htf_map, base_cutoff_idx, htf_cutoff_idx)

    pipeline = feature_pipeline(CAUSAL_FEATURES)
    base_result = pipeline.build(base_map, htf_map)
    altered_result = pipeline.build(altered_base_map, altered_htf_map)

    cutoff_ts = base_map["BTC/USDT"].loc[base_cutoff_idx, "timestamp"]
    for symbol in base_result.feature_map:
        base_features = base_result.feature_map[symbol]
        altered_features = altered_result.feature_map[symbol]
        feature_columns = [
            column
            for column in base_result.feature_columns
            if column in base_features.columns and column in altered_features.columns
        ]
        past_mask = base_features["timestamp"] <= cutoff_ts
        pd.testing.assert_frame_equal(
            base_features.loc[past_mask, feature_columns].reset_index(drop=True),
            altered_features.loc[past_mask, feature_columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            obj=f"{symbol} features before {cutoff_ts}",
        )


def test_feature_pipeline_resolves_requested_feature_blocks():
    request = feature_pipeline(CAUSAL_FEATURES).resolve_request()

    assert set(request.active_features) == set(CAUSAL_FEATURES)
    assert "momentum" in request.active_blocks
    assert "regime" in request.active_blocks
    assert "funding" in request.active_blocks
    assert "premium_index" in request.active_blocks
    assert "open_interest" in request.active_blocks
    assert "htf" in request.active_blocks
    assert "market_context" in request.active_blocks
    assert "interactions" in request.active_blocks
