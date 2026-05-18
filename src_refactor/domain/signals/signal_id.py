from __future__ import annotations

import pandas as pd

from src_refactor.domain.signals.batch_processor import SignalCandidate


def build_signal_id(candidate: SignalCandidate) -> str:
    prediction = candidate.prediction
    timestamp = pd.to_datetime(prediction.timestamp).isoformat()
    return "|".join(
        [
            prediction.model_id,
            prediction.symbol,
            prediction.timeframe,
            timestamp,
            str(candidate.direction),
        ]
    )
