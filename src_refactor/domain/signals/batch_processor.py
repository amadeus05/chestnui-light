from __future__ import annotations

from dataclasses import dataclass

from src_refactor.core.types import Prediction


@dataclass(frozen=True, slots=True)
class SignalProcessingConfig:
    directional_proba_threshold: float = 0.5
    min_signal_gap: float = 0.0
    allow_longs: bool = True
    allow_shorts: bool = True


@dataclass(frozen=True, slots=True)
class SignalCandidate:
    symbol: str
    direction: int
    direction_prob: float
    signal_gap: float
    score: float
    prediction: Prediction


class SignalBatchProcessor:
    def __init__(self, config: SignalProcessingConfig | None = None) -> None:
        self.config = config or SignalProcessingConfig()

    def build_candidates(self, predictions: list[Prediction]) -> list[SignalCandidate]:
        candidates: list[SignalCandidate] = []
        for prediction in predictions:
            candidate = self._candidate_from_prediction(prediction)
            if candidate is not None:
                candidates.append(candidate)
        return sorted(candidates, key=lambda item: (item.score, item.direction_prob), reverse=True)

    def _candidate_from_prediction(self, prediction: Prediction) -> SignalCandidate | None:
        direction = prediction.direction
        p_long = prediction.proba_long
        p_short = prediction.proba_short
        if p_long is not None and p_short is not None:
            direction, direction_prob, signal_gap = self.resolve_directional_signal(p_long, p_short)
        else:
            direction_prob = prediction.confidence
            signal_gap = float(prediction.raw.get("signal_gap", 0.0))

        if direction == 0:
            return None
        if direction == 1 and not self.config.allow_longs:
            return None
        if direction == -1 and not self.config.allow_shorts:
            return None

        return SignalCandidate(
            symbol=prediction.symbol,
            direction=direction,
            direction_prob=direction_prob,
            signal_gap=signal_gap,
            score=self.build_entry_score(direction_prob, signal_gap),
            prediction=prediction,
        )

    def resolve_directional_signal(self, p_long: float, p_short: float) -> tuple[int, float, float]:
        signal_gap = abs(p_long - p_short)
        if (
            p_long >= self.config.directional_proba_threshold
            and (p_long - p_short) >= self.config.min_signal_gap
        ):
            return 1, p_long, signal_gap
        if (
            p_short >= self.config.directional_proba_threshold
            and (p_short - p_long) >= self.config.min_signal_gap
        ):
            return -1, p_short, signal_gap
        return 0, max(p_long, p_short), signal_gap

    def build_entry_score(self, direction_prob: float, signal_gap: float) -> float:
        edge = max(0.0, direction_prob - self.config.directional_proba_threshold)
        return edge * 10 + signal_gap
