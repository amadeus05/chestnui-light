from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src_refactor.core.contracts import PredictionStore
from src_refactor.core.types import Prediction
from src_refactor.infrastructure.predictions.timestamps import canonical_prediction_timestamp


@dataclass(frozen=True, slots=True)
class ParquetPredictionStore(PredictionStore):
    path: Path

    def write(self, predictions: Iterable[Prediction]) -> None:
        rows = [_prediction_to_row(prediction) for prediction in predictions]
        if not rows:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_frame = pd.DataFrame(rows)
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            new_frame = pd.concat([existing, new_frame], ignore_index=True)

        new_frame = new_frame.drop_duplicates(
            subset=["timestamp", "symbol", "timeframe", "model_id", "fold_id"],
            keep="last",
        )
        new_frame = new_frame.sort_values(["timestamp", "symbol", "model_id"]).reset_index(drop=True)
        new_frame.to_parquet(self.path, index=False)

    def read(
        self,
        *,
        model_id: str,
        symbols: tuple[str, ...] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[Prediction]:
        if not self.path.exists():
            return []

        frame = pd.read_parquet(self.path)
        if frame.empty:
            return []

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None)
        mask = frame["model_id"] == model_id
        if symbols:
            mask &= frame["symbol"].isin(symbols)
        if start is not None:
            mask &= frame["timestamp"] >= canonical_prediction_timestamp(start)
        if end is not None:
            mask &= frame["timestamp"] <= canonical_prediction_timestamp(end)

        filtered = frame.loc[mask].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return [_row_to_prediction(row) for _, row in filtered.iterrows()]


def _prediction_to_row(prediction: Prediction) -> dict[str, Any]:
    return {
        "timestamp": canonical_prediction_timestamp(prediction.timestamp),
        "symbol": prediction.symbol,
        "timeframe": prediction.timeframe,
        "model_id": prediction.model_id,
        "direction": int(prediction.direction),
        "confidence": float(prediction.confidence),
        "fold_id": prediction.fold_id,
        "proba_long": prediction.proba_long,
        "proba_short": prediction.proba_short,
        "raw_json": json.dumps(prediction.raw, ensure_ascii=True, default=str),
    }


def _row_to_prediction(row: pd.Series) -> Prediction:
    fold_id = row.get("fold_id")
    return Prediction(
        timestamp=canonical_prediction_timestamp(row["timestamp"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        model_id=str(row["model_id"]),
        direction=int(row["direction"]),
        confidence=float(row["confidence"]),
        fold_id=None if pd.isna(fold_id) else int(fold_id),
        proba_long=_optional_float(row.get("proba_long")),
        proba_short=_optional_float(row.get("proba_short")),
        raw=json.loads(row.get("raw_json") or "{}"),
    )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
