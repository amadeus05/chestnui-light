from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.domain.features.primitives.indicators import rolling_matthews_corrcoef_sign_agreement

BTC_RELATIVE_FEATURE_COLUMNS = [
    "relative_strength_vs_btc_24h",
    "beta_to_btc_24h",
    "residual_return_24h",
    "mcc_sign_agreement_btc_24h",
]


@dataclass(frozen=True, slots=True)
class BtcRelativeFeatureConfig:
    mcc_sign_btc_window: int = 24
    mcc_sign_btc_min_periods: int | None = None

    @property
    def mcc_window(self) -> int:
        return max(2, int(self.mcc_sign_btc_window))

    @property
    def mcc_min_periods(self) -> int:
        if self.mcc_sign_btc_min_periods is not None:
            return max(3, int(self.mcc_sign_btc_min_periods))
        return max(3, self.mcc_window // 2)


def enrich_btc_relative_features(
    feature_map: dict[str, pd.DataFrame],
    btc_symbol: str = "BTC/USDT",
    timestamp_column: str = "timestamp_ms",
    config: BtcRelativeFeatureConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or BtcRelativeFeatureConfig()
    btc_reference = _build_btc_reference(feature_map.get(btc_symbol), timestamp_column)
    return {
        symbol: _enrich_symbol(symbol, frame, btc_reference, btc_symbol, timestamp_column, config)
        for symbol, frame in feature_map.items()
    }


def _build_btc_reference(btc_frame: pd.DataFrame | None, timestamp_column: str) -> pd.DataFrame:
    if btc_frame is None or btc_frame.empty:
        return pd.DataFrame(columns=[timestamp_column, "btc_close", "btc_return_1h_24"])
    return btc_frame[[timestamp_column, "close", "return_1h_24"]].copy().rename(
        columns={"close": "btc_close", "return_1h_24": "btc_return_1h_24"}
    )


def _enrich_symbol(
    symbol: str,
    frame: pd.DataFrame,
    btc_reference: pd.DataFrame,
    btc_symbol: str,
    timestamp_column: str,
    config: BtcRelativeFeatureConfig,
) -> pd.DataFrame:
    output = frame.copy()
    for column in BTC_RELATIVE_FEATURE_COLUMNS:
        output[column] = np.nan

    if "return_1h_24" not in frame.columns:
        raise ValueError("BTC-relative features require 'return_1h_24'.")

    merged = frame[[timestamp_column, "close", "return_1h_24"]].merge(
        btc_reference,
        on=timestamp_column,
        how="left",
    )
    if merged["btc_close"].isna().all():
        return output

    asset_return_1h = np.log(merged["close"] / merged["close"].shift(1))
    btc_return_1h = np.log(merged["btc_close"] / merged["btc_close"].shift(1))
    btc_var_24h = btc_return_1h.rolling(24).var().replace(0, np.nan)
    beta_24h = asset_return_1h.rolling(24).cov(btc_return_1h) / btc_var_24h

    output["relative_strength_vs_btc_24h"] = merged["return_1h_24"] - merged["btc_return_1h_24"]
    output["beta_to_btc_24h"] = beta_24h
    output["residual_return_24h"] = merged["return_1h_24"] - (beta_24h * merged["btc_return_1h_24"])
    output["mcc_sign_agreement_btc_24h"] = rolling_matthews_corrcoef_sign_agreement(
        asset_return_1h,
        btc_return_1h,
        window=config.mcc_window,
        min_periods=config.mcc_min_periods,
    )

    if symbol == btc_symbol:
        output["relative_strength_vs_btc_24h"] = 0.0
        output["beta_to_btc_24h"] = 1.0
        output["residual_return_24h"] = 0.0
        output["mcc_sign_agreement_btc_24h"] = 1.0
    return output
