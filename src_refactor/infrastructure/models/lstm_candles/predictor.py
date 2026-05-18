from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

from src_refactor.core.contracts import ModelPredictor
from src_refactor.core.types import ModelInput, ModelSpec, Prediction
from src_refactor.core.types import LstmCandleWindowInput
from src_refactor.infrastructure.models.lstm_common import LSTMClassifier, LstmTrainingConfig, SequenceStandardizer, predict_lstm_proba


@dataclass(frozen=True, slots=True)
class LstmCandlePredictor(ModelPredictor):
    spec: ModelSpec
    model: Any
    standardizer: SequenceStandardizer | None = None
    training_config: LstmTrainingConfig | None = None

    def predict(self, model_input: ModelInput) -> Prediction:
        if not isinstance(model_input, LstmCandleWindowInput):
            raise TypeError("LstmCandlePredictor expects LstmCandleWindowInput.")
        if self.standardizer is None:
            raise RuntimeError("LSTM candle predictor requires fitted standardizer.")
        config = self.training_config or LstmTrainingConfig.from_metadata(self.spec.metadata)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        proba = predict_lstm_proba(
            model,
            model_input.candles,
            self.standardizer,
            device=device,
            batch_size=config.batch_size,
        )
        if len(proba) == 0:
            raise ValueError("LSTM candle prediction input produced no sequences.")
        p_short = float(proba[-1][0])
        p_long = float(proba[-1][1])
        frame = model_input.metadata.get("frame")
        latest_row = frame.iloc[-1] if isinstance(frame, pd.DataFrame) and not frame.empty else None
        return Prediction(
            timestamp=_resolve_timestamp(latest_row),
            symbol=str(latest_row.get("symbol", "")) if latest_row is not None else "",
            timeframe=self.spec.timeframe,
            model_id=self.spec.model_id,
            direction=0,
            confidence=max(p_long, p_short),
            proba_long=p_long,
            proba_short=p_short,
            raw={
                "signal_gap": abs(p_long - p_short),
                "feature_names": list(model_input.candle_columns),
            },
        )


def build_lstm_candle_model(input_size: int, config: LstmTrainingConfig) -> LSTMClassifier:
    return LSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _resolve_timestamp(row: pd.Series | None) -> pd.Timestamp:
    if row is None:
        return pd.Timestamp.utcnow()
    if "timestamp" in row:
        return pd.to_datetime(row["timestamp"], utc=True)
    if "timestamp_ms" in row:
        return pd.to_datetime(int(row["timestamp_ms"]), unit="ms", utc=True)
    return pd.Timestamp.utcnow()
