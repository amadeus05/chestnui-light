"""
Единая точка входа для расчёта фич по свечам (тот же путь, что ETL, paper, future live).

- Офлайн: ``compute`` → в ETL пишем в БД; бэктест с precomputed только читает готовые строки.
- Онлайн: ``compute`` на окне свечей с биржи, без использования feature-таблицы.

`MasterFeatureBuilder` остаётся реализацией; снаружи коду удобнее знать про один `FeaturePipeline`.
"""
from __future__ import annotations

import pandas as pd

from src.features.master_feature_builder import MasterFeatureBuilder
from src.features.models.feature_pipeline_result import FeaturePipelineResult
from src.features.models.feature_spec import FeatureSpec


class FeaturePipeline:
    def __init__(self, builder: MasterFeatureBuilder | None = None) -> None:
        self._builder = builder or MasterFeatureBuilder()

    def compute(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame],
    ) -> FeaturePipelineResult:
        return self._builder.build(base_candle_map, htf_candle_map)

    def feature_specs(
        self,
        requested_features: set[str] | None = None,
    ) -> dict[str, FeatureSpec]:
        return self._builder.collect_feature_specs(requested_features)


def row_at_timestamp(
    feature_df: pd.DataFrame,
    ts: pd.Timestamp,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame | None:
    """Одна строка фич на метке времени (для согласованного доступа к precomputed/рантайм)."""
    if feature_df is None or feature_df.empty or timestamp_column not in feature_df.columns:
        return None
    row = feature_df[feature_df[timestamp_column] == ts]
    if row.empty:
        return None
    return row.iloc[[-1]].copy()
