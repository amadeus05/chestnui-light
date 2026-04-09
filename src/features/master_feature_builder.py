from __future__ import annotations

import config as cfg
import pandas as pd

from src.features.builders.compression_expansion_feature_builder import CompressionExpansionFeatureBuilder
from src.features.builders.entry_location_feature_builder import EntryLocationFeatureBuilder
from src.features.builders.htf_context_feature_builder import HtfContextFeatureBuilder
from src.features.builders.interaction_feature_builder import InteractionFeatureBuilder
from src.features.builders.lwti_feature_builder import LwtiFeatureBuilder
from src.features.builders.market_context_feature_builder import MarketContextFeatureBuilder
from src.features.builders.regime_feature_builder import RegimeFeatureBuilder
from src.features.builders.session_context_feature_builder import SessionContextFeatureBuilder
from src.features.builders.trend_feature_builder import TrendFeatureBuilder
from src.features.builders.volume_flow_feature_builder import VolumeFlowFeatureBuilder
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_pipeline_result import FeaturePipelineResult
from src.features.models.feature_request import ResolvedFeatureRequest, resolve_feature_request


class MasterFeatureBuilder:
    def __init__(self) -> None:
        self.base_timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
        self.htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        self.main_builders: list[FeatureBuilderContract] = [
            TrendFeatureBuilder(timeframe_label="1h"),
            RegimeFeatureBuilder(timeframe_label="1h"),
            CompressionExpansionFeatureBuilder(),
            EntryLocationFeatureBuilder(),
            InteractionFeatureBuilder(),
            LwtiFeatureBuilder(),
            MarketContextFeatureBuilder(),
            SessionContextFeatureBuilder(),
            VolumeFlowFeatureBuilder(),
        ]
        self.htf_builders: list[FeatureBuilderContract] = [
            HtfContextFeatureBuilder(),
            TrendFeatureBuilder(timeframe_label="4h"),
            RegimeFeatureBuilder(timeframe_label="4h"),
        ]

    def build(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame],
    ) -> FeaturePipelineResult:
        request = self._resolve_request()
        requested_features = set(request.active_features)

        base_feature_map = self._build_symbol_map(
            candle_map=base_candle_map,
            builders=self.main_builders,
            requested_features=requested_features,
            htf_candle_map=htf_candle_map,
        )
        htf_feature_map = self._build_symbol_map(
            candle_map=htf_candle_map,
            builders=self.htf_builders,
            requested_features=requested_features,
        )
        final_feature_map = self._merge_main_and_htf(base_feature_map, htf_feature_map, requested_features)

        return FeaturePipelineResult(
            feature_map=final_feature_map,
            feature_columns=tuple(sorted(requested_features)),
            active_blocks=request.active_blocks,
            profile_name=request.profile,
        )

    def _resolve_request(self) -> ResolvedFeatureRequest:
        block_features = self._collect_block_features()
        raw_request = getattr(cfg, "FEATURE_BUILD_REQUEST", {})
        profile_map = getattr(cfg, "FEATURE_PROFILES", {})
        return resolve_feature_request(raw_request, profile_map, block_features)

    def _collect_block_features(self) -> dict[str, set[str]]:
        block_features: dict[str, set[str]] = {}
        for builder in [*self.main_builders, *self.htf_builders]:
            block_features.setdefault(builder.block_name, set()).update(builder.provides())
        return block_features

    def _build_symbol_map(
        self,
        candle_map: dict[str, pd.DataFrame],
        builders: list[FeatureBuilderContract],
        requested_features: set[str],
        htf_candle_map: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        built_map: dict[str, pd.DataFrame] = {}
        for symbol, source_df in candle_map.items():
            frame = source_df.copy().sort_values("timestamp").reset_index(drop=True)
            output = frame.copy()
            context = FeatureContext(
                frame=frame,
                symbol=symbol,
                base_feature_map=candle_map,
                htf_feature_map=htf_candle_map,
            )
            for builder in builders:
                block_request = builder.provides().intersection(requested_features)
                if not block_request:
                    continue
                feature_block = builder.build(context, block_request)
                output = self._merge_feature_block(output, feature_block)
            built_map[symbol] = output
        return built_map

    def _merge_main_and_htf(
        self,
        base_feature_map: dict[str, pd.DataFrame],
        htf_feature_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        merged_map: dict[str, pd.DataFrame] = {}
        htf_feature_names = set().union(*(builder.provides() for builder in self.htf_builders))
        htf_requested_columns = [column for column in sorted(htf_feature_names) if column in requested_features]

        for symbol, base_df in base_feature_map.items():
            output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
            htf_df = htf_feature_map.get(symbol)
            if htf_df is None or htf_df.empty:
                for column in htf_requested_columns:
                    output[column] = pd.NA
                merged_map[symbol] = output
                continue

            merge_columns = [column for column in htf_requested_columns if column in htf_df.columns]
            base_merge = output.copy()
            base_merge["_merge_ts"] = base_merge["timestamp"] + self._timeframe_to_timedelta(self.base_timeframe)

            htf_merge = htf_df[["timestamp"] + merge_columns].copy()
            # Align HTF features only after the higher-timeframe candle is fully closed.
            htf_merge["_merge_ts"] = htf_merge["timestamp"] + self._timeframe_to_timedelta(self.htf_timeframe)
            merged = pd.merge_asof(
                base_merge.sort_values("_merge_ts").reset_index(drop=True),
                htf_merge[["_merge_ts"] + merge_columns].sort_values("_merge_ts").reset_index(drop=True),
                on="_merge_ts",
                direction="backward",
            )
            merged = merged.drop(columns=["_merge_ts"], errors="ignore")
            for column in htf_requested_columns:
                if column not in merged.columns:
                    merged[column] = pd.NA
            merged_map[symbol] = merged
        return merged_map

    @staticmethod
    def _merge_feature_block(base_df: pd.DataFrame, feature_block: pd.DataFrame) -> pd.DataFrame:
        if feature_block.empty or list(feature_block.columns) == ["timestamp"]:
            return base_df
        extra_columns = [column for column in feature_block.columns if column != "timestamp"]
        return base_df.merge(feature_block[["timestamp"] + extra_columns], on="timestamp", how="left")

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
        amount = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        if unit == "m":
            return pd.to_timedelta(amount, unit="m")
        if unit == "h":
            return pd.to_timedelta(amount, unit="h")
        if unit == "d":
            return pd.to_timedelta(amount, unit="d")
        if unit == "w":
            return pd.to_timedelta(amount * 7, unit="d")
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
