from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src_refactor.domain.features.builders.base.derivatives import (
    DerivativesFeatureConfig,
    FUNDING_FEATURE_COLUMNS,
    OPEN_INTEREST_FEATURE_COLUMNS,
    PREMIUM_FEATURE_COLUMNS,
    build_funding_frame,
    build_open_interest_frame,
    build_premium_frame,
)
from src_refactor.domain.features.builders.base.momentum import (
    MOMENTUM_FEATURE_COLUMNS,
    MomentumFeatureConfig,
    build_momentum_frame,
)
from src_refactor.domain.features.builders.base.regime import REGIME_FEATURE_COLUMNS, RegimeFeatureConfig, build_regime_frame
from src_refactor.domain.features.builders.base.structure import STRUCTURE_FEATURE_COLUMNS, build_structure_frame
from src_refactor.domain.features.builders.base.time_context import TIME_CONTEXT_FEATURE_COLUMNS, build_time_context_frame
from src_refactor.domain.features.builders.cross_symbol.btc_relative import (
    BTC_RELATIVE_FEATURE_COLUMNS,
    BtcRelativeFeatureConfig,
    enrich_btc_relative_features,
)
from src_refactor.domain.features.builders.cross_symbol.cross_symbol import (
    BASE_CROSS_SECTIONAL_COLUMNS,
    BASE_MARKET_CONTEXT_COLUMNS,
    HTF_CROSS_SECTIONAL_COLUMNS,
    HTF_MARKET_CONTEXT_COLUMNS,
    enrich_base_cross_sectional_rank,
    enrich_base_market_context,
    enrich_htf_cross_sectional_rank,
    enrich_htf_market_context,
)
from src_refactor.domain.features.builders.htf.htf import HTF_FEATURE_COLUMNS, HtfFeatureConfig, build_htf_frame
from src_refactor.domain.features.builders.post_merge.interactions import (
    INTERACTION_FEATURE_COLUMNS,
    InteractionFeatureConfig,
    build_interaction_frame,
)
from src_refactor.domain.features.pipeline.feature_request import ResolvedFeatureRequest, resolve_feature_request
from src_refactor.domain.features.pipeline.timeframe_merge import merge_main_and_htf_features


@dataclass(frozen=True, slots=True)
class FeaturePipelineConfig:
    raw_request: dict | None = None
    profile_map: dict[str, object] | None = None
    momentum: MomentumFeatureConfig = field(default_factory=MomentumFeatureConfig)
    regime: RegimeFeatureConfig = field(default_factory=RegimeFeatureConfig)
    derivatives: DerivativesFeatureConfig = field(default_factory=DerivativesFeatureConfig)
    htf: HtfFeatureConfig = field(default_factory=HtfFeatureConfig)
    btc_relative: BtcRelativeFeatureConfig = field(default_factory=BtcRelativeFeatureConfig)
    interactions: InteractionFeatureConfig = field(default_factory=InteractionFeatureConfig)


@dataclass(frozen=True, slots=True)
class FeaturePipelineResult:
    feature_map: dict[str, pd.DataFrame]
    feature_columns: tuple[str, ...]
    active_blocks: tuple[str, ...]
    profile_name: str


class FeaturePipeline:
    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()

    def build(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame] | None = None,
    ) -> FeaturePipelineResult:
        request = self.resolve_request()
        requested_features = set(request.active_features)
        base_feature_map = self._build_base_feature_map(base_candle_map, requested_features)
        htf_feature_map = self._build_htf_feature_map(htf_candle_map or {}, requested_features)

        if requested_features.intersection(BTC_RELATIVE_FEATURE_COLUMNS):
            base_feature_map = enrich_btc_relative_features(base_feature_map, config=self.config.btc_relative)
        if requested_features.intersection(BASE_CROSS_SECTIONAL_COLUMNS):
            base_feature_map = enrich_base_cross_sectional_rank(base_feature_map)
        if requested_features.intersection(BASE_MARKET_CONTEXT_COLUMNS):
            base_feature_map = enrich_base_market_context(base_feature_map)
        if requested_features.intersection(HTF_CROSS_SECTIONAL_COLUMNS):
            htf_feature_map = enrich_htf_cross_sectional_rank(htf_feature_map)
        if requested_features.intersection(HTF_MARKET_CONTEXT_COLUMNS):
            htf_feature_map = enrich_htf_market_context(htf_feature_map)

        merged_map = self._merge_htf(base_feature_map, htf_feature_map, requested_features)
        if requested_features.intersection(INTERACTION_FEATURE_COLUMNS):
            merged_map = {
                symbol: self._merge_block(frame, build_interaction_frame(frame, self.config.interactions))
                for symbol, frame in merged_map.items()
            }

        final_map = {
            symbol: self._select_output_columns(frame, requested_features)
            for symbol, frame in merged_map.items()
        }
        return FeaturePipelineResult(
            feature_map=final_map,
            feature_columns=tuple(sorted(requested_features)),
            active_blocks=request.active_blocks,
            profile_name=request.profile,
        )

    def build_for_symbol(
        self,
        symbol: str,
        base_candles: pd.DataFrame,
        htf_candles: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        result = self.build({symbol: base_candles}, {symbol: htf_candles} if htf_candles is not None else None)
        return result.feature_map[symbol]

    def build_latest_for_symbol(
        self,
        symbol: str,
        base_candles: pd.DataFrame,
        htf_candles: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.build_for_symbol(symbol, base_candles, htf_candles).tail(1).copy()

    @staticmethod
    def block_features() -> dict[str, set[str]]:
        return {
            "momentum": set(MOMENTUM_FEATURE_COLUMNS),
            "regime": set(REGIME_FEATURE_COLUMNS),
            "structure": set(STRUCTURE_FEATURE_COLUMNS),
            "time_context": set(TIME_CONTEXT_FEATURE_COLUMNS),
            "funding": set(FUNDING_FEATURE_COLUMNS),
            "premium_index": set(PREMIUM_FEATURE_COLUMNS),
            "open_interest": set(OPEN_INTEREST_FEATURE_COLUMNS),
            "htf": set(HTF_FEATURE_COLUMNS),
            "btc_relative": set(BTC_RELATIVE_FEATURE_COLUMNS),
            "cross_sectional": set(BASE_CROSS_SECTIONAL_COLUMNS) | set(HTF_CROSS_SECTIONAL_COLUMNS),
            "market_context": set(BASE_MARKET_CONTEXT_COLUMNS) | set(HTF_MARKET_CONTEXT_COLUMNS),
            "interactions": set(INTERACTION_FEATURE_COLUMNS),
        }

    def resolve_request(self) -> ResolvedFeatureRequest:
        return resolve_feature_request(self.config.raw_request, self.config.profile_map, self.block_features())

    def _build_base_feature_map(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for symbol, source in base_candle_map.items():
            frame = self._normalize_frame(source)
            if requested_features.intersection(MOMENTUM_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_momentum_frame(frame, self.config.momentum))
            if requested_features.intersection(REGIME_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_regime_frame(frame, self.config.regime))
            if requested_features.intersection(STRUCTURE_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_structure_frame(frame))
            if requested_features.intersection(TIME_CONTEXT_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_time_context_frame(frame["timestamp_ms"]))
            if requested_features.intersection(FUNDING_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_funding_frame(frame, self.config.derivatives))
            if requested_features.intersection(PREMIUM_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_premium_frame(frame, self.config.derivatives))
            if requested_features.intersection(OPEN_INTEREST_FEATURE_COLUMNS):
                frame = self._merge_block(frame, build_open_interest_frame(frame, self.config.derivatives))
            output[symbol] = frame
        return output

    def _build_htf_feature_map(
        self,
        htf_candle_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        if not requested_features.intersection(
            set(HTF_FEATURE_COLUMNS) | set(HTF_CROSS_SECTIONAL_COLUMNS) | set(HTF_MARKET_CONTEXT_COLUMNS)
        ):
            return {}
        return {
            symbol: self._merge_block(
                self._normalize_frame(source),
                build_htf_frame(self._normalize_frame(source), self.config.htf),
            )
            for symbol, source in htf_candle_map.items()
        }

    def _merge_htf(
        self,
        base_feature_map: dict[str, pd.DataFrame],
        htf_feature_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        htf_requested_columns = [
            column
            for column in [*HTF_FEATURE_COLUMNS, *HTF_CROSS_SECTIONAL_COLUMNS, *HTF_MARKET_CONTEXT_COLUMNS]
            if column in requested_features
        ]
        if not htf_requested_columns:
            return base_feature_map
        return {
            symbol: merge_main_and_htf_features(frame, htf_feature_map.get(symbol), htf_requested_columns)
            for symbol, frame in base_feature_map.items()
        }

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if "timestamp_ms" not in output.columns:
            if "timestamp" not in output.columns:
                raise ValueError("Feature pipeline input frame is missing column: timestamp_ms or timestamp")
            output["timestamp_ms"] = pd.to_datetime(output["timestamp"], utc=True).astype("int64") // 1_000_000
        output = output.sort_values("timestamp_ms").reset_index(drop=True)
        output["timestamp_ms"] = output["timestamp_ms"].astype("int64")
        return output

    @staticmethod
    def _merge_block(frame: pd.DataFrame, feature_block: pd.DataFrame) -> pd.DataFrame:
        if feature_block.empty or list(feature_block.columns) == ["timestamp_ms"]:
            return frame
        extra_columns = [column for column in feature_block.columns if column != "timestamp_ms"]
        deduped_block = feature_block.loc[:, ["timestamp_ms", *extra_columns]]
        output = frame.drop(columns=[column for column in extra_columns if column in frame.columns])
        return output.merge(deduped_block, on="timestamp_ms", how="left")

    @staticmethod
    def _select_output_columns(frame: pd.DataFrame, requested_features: set[str]) -> pd.DataFrame:
        raw_columns = [
            column
            for column in ("timestamp_ms", "timestamp", "open", "high", "low", "close", "volume", "quote_volume")
            if column in frame.columns
        ]
        feature_columns = [column for column in sorted(requested_features) if column in frame.columns]
        return frame[[*raw_columns, *feature_columns]].copy()
