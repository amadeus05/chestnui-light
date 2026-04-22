from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.application.candidate_builder import ExecutionCandidateBuilder
from src.application.entry_candidate_resolution import resolve_and_build_entry_candidate
from src.domain.risk.risk_manager import RiskPolicy, effective_risk_per_trade as resolve_effective_risk
from src.backtest.feature_utils import (
    apply_feature_clip_bounds,
    get_barrier_pcts,
    get_feature_batch_precomputed,
    is_candidate_event,
    normalize_features_for_model,
)
from src.backtest.ohlc_access import get_feature_row_precomputed
from src.domain.strategy.signal_engine import SignalEngine
from src.execution import EntryCandidate, ExecutionEngine, PortfolioState, RuntimePosition


@dataclass
class BacktestCandidateConfig:
    allow_longs: bool
    allow_shorts: bool
    backtest_realtime_features: bool
    risk_per_trade: float
    reduce_risk_after_consecutive_losses: int
    reduced_risk_per_trade: float


class BacktestCandidateBuilder:
    """
    Сборка EntryCandidate для бэктеста — логика перенесена из bt._build_entry_candidates.
    """

    def __init__(
        self,
        *,
        config: BacktestCandidateConfig,
        execution_engine: ExecutionEngine,
        feature_names: list[str],
        symbol_categories: list[str] | None,
        clip_bounds: dict,
        event_filter_config: dict,
        all_features: dict[str, pd.DataFrame],
        all_features_prepared: dict[str, pd.DataFrame],
        prediction_lookup: dict,
        using_external_predictions: bool,
        predictions_have_barriers: bool,
        model: Any,
        runtime_predictor: Any,
        signal_engine: SignalEngine | None,
        candidate_builder: ExecutionCandidateBuilder | None,
        runtime_feature_runtime_service: Any,
        build_feature_row_at_time: Callable[..., pd.DataFrame | None] | None,
    ) -> None:
        self._cfg = config
        self._execution_engine = execution_engine
        self._feature_names = feature_names
        self._symbol_categories = symbol_categories
        self._clip_bounds = clip_bounds
        self._event_filter_config = event_filter_config
        self._all_features = all_features
        self._all_features_prepared = all_features_prepared
        self._prediction_lookup = prediction_lookup
        self._using_external_predictions = using_external_predictions
        self._predictions_have_barriers = predictions_have_barriers
        self._model = model
        self._runtime_predictor = runtime_predictor
        self._signal_engine = signal_engine
        self._candidate_builder = candidate_builder
        self._feature_runtime = runtime_feature_runtime_service
        self._build_feature_row_at_time = build_feature_row_at_time

    def build(
        self,
        i: int,
        current_ts: pd.Timestamp,
        next_ts: pd.Timestamp,
        market_batch: dict[str, dict],
        portfolio_state: PortfolioState,
        positions: dict[str, RuntimePosition | None],
        stop_cooldown_until_index: dict[str, int],
        consecutive_loss_count: int,
    ) -> list[EntryCandidate]:
        risk_policy = RiskPolicy(
            base_risk_per_trade=self._cfg.risk_per_trade,
            reduce_after_consecutive_losses=self._cfg.reduce_risk_after_consecutive_losses,
            reduced_risk_per_trade=self._cfg.reduced_risk_per_trade,
        )
        effective_risk = resolve_effective_risk(risk_policy, consecutive_loss_count)

        snapshot_balance = portfolio_state.balance
        entry_candidates: list[EntryCandidate] = []

        if self._cfg.backtest_realtime_features:
            self._fill_realtime_branch(
                i=i,
                next_ts=next_ts,
                market_batch=market_batch,
                positions=positions,
                stop_cooldown_until_index=stop_cooldown_until_index,
                snapshot_balance=snapshot_balance,
                effective_risk_per_trade=effective_risk,
                entry_candidates=entry_candidates,
            )
        else:
            self._fill_precomputed_branch(
                i=i,
                current_ts=current_ts,
                market_batch=market_batch,
                positions=positions,
                stop_cooldown_until_index=stop_cooldown_until_index,
                snapshot_balance=snapshot_balance,
                effective_risk_per_trade=effective_risk,
                entry_candidates=entry_candidates,
            )

        return entry_candidates

    def _fill_realtime_branch(
        self,
        *,
        i: int,
        next_ts: pd.Timestamp,
        market_batch: dict[str, dict],
        positions: dict[str, RuntimePosition | None],
        stop_cooldown_until_index: dict[str, int],
        snapshot_balance: float,
        effective_risk_per_trade: float,
        entry_candidates: list[EntryCandidate],
    ) -> None:
        builder_fn = self._build_feature_row_at_time
        if builder_fn is None:
            return

        for sym, ctx in market_batch.items():
            if positions[sym] is not None:
                continue
            if i < stop_cooldown_until_index.get(sym, -1):
                continue

            latest_row = builder_fn(
                symbol=sym,
                main_df=ctx["main"],
                htf_df=ctx["htf"],
                analysis_ts=next_ts,
                feature_names=self._feature_names,
                symbol_categories=self._symbol_categories,
            )

            if latest_row is None or latest_row.empty:
                continue

            current_features = normalize_features_for_model(
                latest_row,
                self._feature_names,
                symbol_categories=self._symbol_categories,
            )
            current_features = apply_feature_clip_bounds(current_features, self._clip_bounds)
            if not is_candidate_event(latest_row, self._event_filter_config):
                continue
            stop_pct, take_pct = get_barrier_pcts(latest_row)
            if stop_pct is None or take_pct is None:
                continue

            active_predictor = self._runtime_predictor or self._model
            proba = active_predictor.predict_proba(current_features)[0]
            p_short = float(proba[0])
            p_long = float(proba[1])

            self._append_from_proba(
                sym=str(sym),
                ctx=ctx,
                p_long=p_long,
                p_short=p_short,
                stop_pct=float(stop_pct),
                take_pct=float(take_pct),
                snapshot_balance=snapshot_balance,
                effective_risk_per_trade=effective_risk_per_trade,
                entry_candidates=entry_candidates,
            )

    def _fill_precomputed_branch(
        self,
        *,
        i: int,
        current_ts: pd.Timestamp,
        market_batch: dict[str, dict],
        positions: dict[str, RuntimePosition | None],
        stop_cooldown_until_index: dict[str, int],
        snapshot_balance: float,
        effective_risk_per_trade: float,
        entry_candidates: list[EntryCandidate],
    ) -> None:
        candidate_symbols = [
            sym
            for sym in market_batch
            if (
                positions[sym] is None
                and (sym in self._all_features_prepared or (self._using_external_predictions and self._predictions_have_barriers))
                and i >= stop_cooldown_until_index.get(sym, -1)
            )
        ]
        if self._using_external_predictions and self._predictions_have_barriers:
            batch_symbols = []
            batch_proba = []
            barrier_rows = {}
            for sym in candidate_symbols:
                prediction_record = self._prediction_lookup.get((current_ts, sym))
                if prediction_record is None:
                    continue
                stop_pct = prediction_record.get("barrier_stop_pct")
                take_pct = prediction_record.get("barrier_take_pct")
                if stop_pct is None or take_pct is None:
                    continue
                batch_symbols.append(sym)
                batch_proba.append((float(prediction_record["p_short"]), float(prediction_record["p_long"])))
                barrier_rows[sym] = pd.DataFrame(
                    [
                        {
                            "timestamp": current_ts,
                            "barrier_stop_pct": float(stop_pct),
                            "barrier_take_pct": float(take_pct),
                        }
                    ]
                )
            batch_features = pd.DataFrame(index=range(len(batch_symbols)))
        else:
            if self._feature_runtime is not None:
                batch_symbols, batch_features = self._feature_runtime.get_feature_batch_precomputed(
                    feature_store=self._all_features_prepared,
                    symbols=candidate_symbols,
                    ts=current_ts,
                )
            else:
                batch_symbols, batch_features = get_feature_batch_precomputed(
                    self._all_features_prepared,
                    candidate_symbols,
                    current_ts,
                )
            barrier_rows = {}

        has_batch = bool(batch_symbols) if self._using_external_predictions and self._predictions_have_barriers else not batch_features.empty
        if not has_batch:
            return

        if self._using_external_predictions and not self._predictions_have_barriers:
            batch_proba = []
            resolved_symbols = []
            for sym in batch_symbols:
                prediction_record = self._prediction_lookup.get((current_ts, sym))
                if prediction_record is None:
                    continue
                batch_proba.append(
                    (
                        float(prediction_record["p_short"]),
                        float(prediction_record["p_long"]),
                    )
                )
                resolved_symbols.append(sym)
            batch_symbols = resolved_symbols
        elif not self._using_external_predictions:
            active_predictor = self._runtime_predictor or self._model
            batch_proba = active_predictor.predict_proba(batch_features[self._feature_names])

        for sym, proba in zip(batch_symbols, batch_proba):
            ctx = market_batch[sym]
            if self._using_external_predictions and self._predictions_have_barriers:
                feature_row = barrier_rows.get(sym)
            else:
                feature_row = get_feature_row_precomputed(
                    self._all_features[sym],
                    current_ts,
                    self._feature_names + ["barrier_stop_pct", "barrier_take_pct"],
                )
            stop_pct, take_pct = get_barrier_pcts(feature_row)
            if stop_pct is None or take_pct is None:
                continue
            if (
                not (self._using_external_predictions and self._predictions_have_barriers)
                and not is_candidate_event(feature_row, self._event_filter_config)
            ):
                continue

            p_short = float(proba[0])
            p_long = float(proba[1])

            cand, _skip = resolve_and_build_entry_candidate(
                self._execution_engine,
                symbol=str(sym),
                next_open=float(ctx["next_open"]),
                stop_pct=float(stop_pct),
                take_pct=float(take_pct),
                p_long=p_long,
                p_short=p_short,
                balance=float(snapshot_balance),
                risk_per_trade=float(effective_risk_per_trade),
                allow_longs=self._cfg.allow_longs,
                allow_shorts=self._cfg.allow_shorts,
                signal_engine=self._signal_engine,
                candidate_builder=self._candidate_builder,
            )
            if cand is not None:
                entry_candidates.append(cand)
