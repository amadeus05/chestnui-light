from __future__ import annotations

from src.application.candidate_builder import CandidateInput, ExecutionCandidateBuilder
from src.domain.strategy.signal_engine import SignalEngine
from src.execution import EntryCandidate, ExecutionEngine


def resolve_and_build_entry_candidate(
    execution_engine: ExecutionEngine,
    *,
    symbol: str,
    next_open: float,
    stop_pct: float,
    take_pct: float,
    p_long: float,
    p_short: float,
    balance: float,
    risk_per_trade: float | None,
    allow_longs: bool,
    allow_shorts: bool,
    signal_engine: SignalEngine | None,
    candidate_builder: ExecutionCandidateBuilder | None,
) -> tuple[EntryCandidate | None, str | None]:
    """
    Единая ветвь «вероятности → направление → EntryCandidate», как в бэктесте и paper.

    Возвращает (кандидат, None) или (None, ключ причины пропуска для счётчиков paper).
    """
    eff_risk = float(risk_per_trade) if risk_per_trade is not None else execution_engine.settings.risk_per_trade
    min_notional = execution_engine.settings.min_position_notional

    if signal_engine is not None and candidate_builder is not None:
        decision = signal_engine.resolve(p_long=float(p_long), p_short=float(p_short))
        if decision.is_flat:
            return None, "no_directional_signal"
        if decision.is_long and not allow_longs:
            return None, "longs_disabled"
        if decision.is_short and not allow_shorts:
            return None, "shorts_disabled"
        candidate = candidate_builder.build(
            candidate_input=CandidateInput(
                symbol=str(symbol),
                next_open=float(next_open),
                stop_pct=float(stop_pct),
                take_pct=float(take_pct),
                p_long=float(p_long),
                p_short=float(p_short),
            ),
            signal_decision=decision,
            balance=float(balance),
            risk_per_trade=eff_risk,
        )
        if candidate is None:
            return None, "position_too_small"
        return candidate, None

    signal, direction_prob, signal_gap = execution_engine.resolve_directional_signal(p_long, p_short)
    if signal == 0:
        return None, "no_directional_signal"
    if signal == 1 and not allow_longs:
        return None, "longs_disabled"
    if signal == -1 and not allow_shorts:
        return None, "shorts_disabled"

    entry_price = execution_engine.apply_entry_price_slippage(next_open, signal)
    position_notional, required_margin = execution_engine.calculate_position_size(
        balance=balance,
        stop_pct=stop_pct,
        risk_per_trade=eff_risk,
    )
    if position_notional < min_notional or required_margin <= 0:
        return None, "position_too_small"

    return (
        EntryCandidate(
            symbol=str(symbol),
            direction=int(signal),
            entry_price=float(entry_price),
            position_notional=float(position_notional),
            required_margin=float(required_margin),
            stop_pct=float(stop_pct),
            take_pct=float(take_pct),
            p_long=float(p_long),
            p_short=float(p_short),
            direction_prob=float(direction_prob),
            signal_gap=float(signal_gap),
            score=float(execution_engine.build_entry_score(direction_prob, signal_gap)),
        ),
        None,
    )
