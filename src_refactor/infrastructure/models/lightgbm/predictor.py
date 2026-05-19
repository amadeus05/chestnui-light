from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src_refactor.core.contracts import ModelPredictor
from src_refactor.core.types import ModelInput, ModelSpec, Prediction
from src_refactor.core.types import LightGbmInput


@dataclass(frozen=True, slots=True)
class LightGbmPredictor(ModelPredictor):
    spec: ModelSpec
    model: Any

    def predict(self, model_input: ModelInput) -> Prediction:
        if not isinstance(model_input, LightGbmInput):
            raise TypeError("LightGbmPredictor expects LightGbmInput.")
        if model_input.features.empty:
            raise ValueError("LightGBM input features must not be empty.")

        proba = self.model.predict_proba(model_input.features)
        p_short = float(proba[-1][0])
        p_long = float(proba[-1][1])
        frame = model_input.metadata.get("frame")
        latest_row = frame.iloc[-1] if isinstance(frame, pd.DataFrame) and not frame.empty else None
        timestamp = _resolve_timestamp(latest_row)
        symbol = str(latest_row.get("symbol", "")) if latest_row is not None else ""
        signal_gap = abs(p_long - p_short)
        return Prediction(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=self.spec.timeframe,
            model_id=self.spec.model_id,
            direction=0,
            confidence=max(p_long, p_short),
            proba_long=p_long,
            proba_short=p_short,
            signal_gap=signal_gap,
            raw={
                "feature_names": list(model_input.feature_names),
            },
        )


def _resolve_timestamp(row: pd.Series | None) -> pd.Timestamp:
    if row is None:
        return pd.Timestamp.utcnow()
    if "timestamp" in row:
        return pd.to_datetime(row["timestamp"], utc=True)
    if "timestamp_ms" in row:
        return pd.to_datetime(int(row["timestamp_ms"]), unit="ms", utc=True)
    return pd.Timestamp.utcnow()


def empty_lightgbm_prediction(spec: ModelSpec, *, symbol: str = "") -> Prediction:
    return Prediction(
        timestamp=pd.Timestamp.utcnow(),
        symbol=symbol,
        timeframe=spec.timeframe,
        model_id=spec.model_id,
        direction=0,
        confidence=0.0,
    )
