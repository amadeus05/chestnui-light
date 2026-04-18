from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, safe_ratio
from src.features.models.feature_context import FeatureContext


class RegimeFeatureBuilder(FeatureBuilderContract):
    block_name = "regime"

    def provides(self) -> set[str]:
        return {
            "realized_vol_1h",
            "atr_ratio_1h",
            "volatility_regime_change_1h",
            "range_compression_1h",
            "volatility_acceleration_1h",
            "volume_24h",
            "volume_ratio_1h",
            "volume_zscore_1h",
            "dollar_volume_zscore_1h",
            # Новые признаки для regime detection - 2026-04-05
            "vol_of_vol_1h",  # волатильность волатильности (2-я производная)
            "realized_vol_vs_ema",  # отклонение от EMA волатильности
            "volatility_regime_stability",  # стабильность текущего режима
            "high_vol_stress_indicator",  # индикатор стресса высокой волатильности
            "vol_regime_classification",  # классификация режима: 0=low, 1=normal, 2=high
        }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        realized_vol_window = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_1H", 24)))
        range_short_window = max(2, int(getattr(cfg, "RANGE_COMPRESSION_SHORT_WINDOW_1H", 12)))
        range_long_window = max(range_short_window + 1, int(getattr(cfg, "RANGE_COMPRESSION_LONG_WINDOW_1H", 48)))
        volume_ratio_window = max(2, int(getattr(cfg, "VOLUME_RATIO_WINDOW_1H", 24)))
        volume_zscore_window = max(24, int(getattr(cfg, "VOLUME_ZSCORE_WINDOW_1H", 24 * 7)))
        volume_24h_window = max(2, int(getattr(cfg, "VOLUME_24H_WINDOW_1H", 24)))

        if "realized_vol_1h" in active:
            log_return_1h_1 = np.log(close / close.shift(1))
            output["realized_vol_1h"] = log_return_1h_1.rolling(realized_vol_window).std()

        if {"atr_ratio_1h", "volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            if "atr_ratio_1h" in active:
                atr_100 = context.indicator_cache.get_or_create(
                    "atr_100",
                    lambda: compute_atr(high, low, close, length=100),
                )
                output["atr_ratio_1h"] = safe_ratio(atr_14, atr_100)

            if {"volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
                atr_6 = context.indicator_cache.get_or_create(
                    "atr_6",
                    lambda: compute_atr(high, low, close, length=6),
                )
                atr_48 = context.indicator_cache.get_or_create(
                    "atr_48",
                    lambda: compute_atr(high, low, close, length=48),
                )
                regime_change = context.indicator_cache.get_or_create(
                    "volatility_regime_change_1h",
                    lambda: safe_ratio(atr_6, atr_48),
                )
                if "volatility_regime_change_1h" in active:
                    output["volatility_regime_change_1h"] = regime_change
                if "volatility_acceleration_1h" in active:
                    output["volatility_acceleration_1h"] = regime_change - regime_change.shift(3)

        if "range_compression_1h" in active:
            range_short = high.rolling(range_short_window).max() - low.rolling(range_short_window).min()
            range_long = high.rolling(range_long_window).max() - low.rolling(range_long_window).min()
            output["range_compression_1h"] = safe_ratio(range_short, range_long)

        if "volume_24h" in active:
            output["volume_24h"] = volume.rolling(volume_24h_window).sum()

        volume_request = {"volume_ratio_1h", "volume_zscore_1h", "dollar_volume_zscore_1h"}
        if volume_request.intersection(active):
            if {"volume_ratio_1h", "dollar_volume_zscore_1h"}.intersection(active):
                volume_mean = volume.rolling(volume_ratio_window).mean()
            if "volume_ratio_1h" in active:
                output["volume_ratio_1h"] = safe_ratio(volume, volume_mean)

            if {"volume_zscore_1h", "dollar_volume_zscore_1h"}.intersection(active):
                volume_roll_mean = volume.rolling(volume_zscore_window).mean()
                volume_roll_std = volume.rolling(volume_zscore_window).std().replace(0, np.nan)
                dollar_volume = np.log1p((close * volume).clip(lower=0))
                dollar_roll_mean = dollar_volume.rolling(volume_zscore_window).mean()
                dollar_roll_std = dollar_volume.rolling(volume_zscore_window).std().replace(0, np.nan)
            if "volume_zscore_1h" in active:
                output["volume_zscore_1h"] = (volume - volume_roll_mean) / volume_roll_std
            if "dollar_volume_zscore_1h" in active:
                output["dollar_volume_zscore_1h"] = (dollar_volume - dollar_roll_mean) / dollar_roll_std

        # ═════════════════════════════════════════════════════════════════
        # Новые признаки для Regime Detection - 2026-04-05
        # Помогают обнаружить структурные сдвиги в рынке (как в Fold 5)
        # ═════════════════════════════════════════════════════════════════
        regime_request = {
            "vol_of_vol_1h",
            "realized_vol_vs_ema",
            "volatility_regime_stability",
            "high_vol_stress_indicator",
            "vol_regime_classification",
        }
        if regime_request.intersection(active):
            # Базовая волатильность должна быть рассчитана
            rvol = output.get("realized_vol_1h")
            if rvol is None and "realized_vol_1h" in active:
                log_return_1h_1 = np.log(close / close.shift(1))
                rvol = log_return_1h_1.rolling(realized_vol_window).std()

            if rvol is not None:
                # 1. Vol of Vol - волатильность волатильности (скачет ли волатильность)
                if "vol_of_vol_1h" in active:
                    output["vol_of_vol_1h"] = rvol.rolling(12).std()

                # 2. Realized Vol vs EMA - отклонение от тренда волатильности
                if "realized_vol_vs_ema" in active:
                    rvol_ema = rvol.ewm(span=48, adjust=False).mean()
                    output["realized_vol_vs_ema"] = safe_ratio(rvol - rvol_ema, rvol_ema)

                # 3. Volatility Regime Stability - сколько баров держится текущий режим
                if "volatility_regime_stability" in active:
                    # Определяем режим по отношению к долгосрочной средней
                    rvol_long_mean = rvol.rolling(96).mean()
                    regime_high = rvol > (rvol_long_mean * 1.2)
                    regime_low = rvol < (rvol_long_mean * 0.8)
                    regime_label = pd.Series(1, index=frame.index, dtype=int)
                    regime_label.loc[regime_low] = 0
                    regime_label.loc[regime_high] = 2

                    # Causal stability: bars elapsed in the current regime up to t.
                    regime_groups = (regime_label != regime_label.shift(1)).cumsum()
                    regime_age = regime_label.groupby(regime_groups).cumcount() + 1
                    output["volatility_regime_stability"] = regime_age.clip(0, 48) / 48.0

                # 4. High Vol Stress Indicator - комбинация высокой волы и расширяющегося диапазона
                if "high_vol_stress_indicator" in active:
                    # Проверяем: высокая волатильность + расширяющийся ATR
                    atr_14 = context.indicator_cache.get_or_create(
                        "atr_14",
                        lambda: compute_atr(high, low, close, length=14),
                    )
                    atr_48 = context.indicator_cache.get_or_create(
                        "atr_48",
                        lambda: compute_atr(high, low, close, length=48),
                    )
                    atr_expanding = atr_14 > atr_48 * 1.1
                    vol_high = rvol > rvol.rolling(96).quantile(0.75)
                    output["high_vol_stress_indicator"] = (atr_expanding & vol_high).astype(float)

                # 5. Vol Regime Classification - категориальный признак (0=low, 1=normal, 2=high)
                if "vol_regime_classification" in active:
                    rvol_33 = rvol.rolling(96).quantile(0.33)
                    rvol_67 = rvol.rolling(96).quantile(0.67)
                    output["vol_regime_classification"] = (
                        (rvol > rvol_33).astype(int) + (rvol > rvol_67).astype(int)
                    )

        return output
