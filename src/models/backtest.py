from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import bt
import config as cfg
from src.models.artifacts import ArtifactStore
from src.models.contracts import ModelSpec


@dataclass(frozen=True, slots=True)
class PredictionBacktestRequest:
    predictions_path: Path | None = None
    metadata_path: Path | None = None
    chart_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None


class PredictionBacktestRunner:
    """Replay backtest for saved OOS predictions in the unified artifact layout."""

    def __init__(self, spec: ModelSpec, artifact_store: ArtifactStore | None = None):
        self.spec = spec
        self.artifact_store = artifact_store or ArtifactStore()

    def run(self, request: PredictionBacktestRequest) -> None:
        paths = self.artifact_store.paths_for(self.spec)
        metadata_path = request.metadata_path or paths.metadata
        predictions_path = request.predictions_path or paths.predictions
        chart_path = request.chart_path or cfg.BACKTEST_CHARTS_DIR / self.spec.default_chart_name

        metadata = self.artifact_store.read_json(metadata_path)
        predictions = self.load_predictions(predictions_path)
        predictions = self.filter_predictions(predictions, request.start_date, request.end_date)

        bt.backtest(
            features_meta=self.to_backtest_features_meta(metadata, predictions),
            predictions=predictions,
            equity_curve_path=chart_path,
            result_title=f"{self.spec.display_name.upper()} WALK-FORWARD OOS BACKTEST",
        )

    @staticmethod
    def load_predictions(path: Path) -> pd.DataFrame:
        predictions = pd.read_csv(path, parse_dates=["timestamp"])
        required_columns = {"timestamp", "symbol", "p_short", "p_long"}
        missing = sorted(required_columns - set(predictions.columns))
        if missing:
            raise ValueError(f"Missing required prediction columns: {missing}")
        return predictions

    @staticmethod
    def filter_predictions(
        predictions: pd.DataFrame,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        filtered = predictions.copy()
        if start_date:
            filtered = filtered.loc[filtered["timestamp"] >= pd.to_datetime(start_date, errors="raise")].copy()
        if end_date:
            filtered = filtered.loc[filtered["timestamp"] <= pd.to_datetime(end_date, errors="raise")].copy()
        if filtered.empty:
            raise ValueError("No predictions remain after date filters.")
        return filtered

    @staticmethod
    def to_backtest_features_meta(metadata: dict, predictions: pd.DataFrame) -> dict:
        train_period = metadata.get("train_period")
        if train_period is None:
            train_period = {
                "start": str(pd.to_datetime(predictions["timestamp"]).min()),
                "end": str(pd.to_datetime(predictions["timestamp"]).max()),
            }

        return {
            "feature_columns": list(metadata.get("feature_columns") or []),
            "label_mapping": metadata.get("label_mapping") or {"short": 0, "long": 1},
            "inverse_label_mapping": metadata.get("inverse_label_mapping") or {"0": -1, "1": 1},
            "symbols": list(metadata.get("symbols") or sorted(predictions["symbol"].astype(str).unique().tolist())),
            "rows": int(len(predictions)),
            "task_type": f"{metadata.get('model_key', 'model')}_walk_forward_oos_replay",
            "train_period": train_period,
            "event_filter": metadata.get("event_filter") or {},
            "feature_clip": metadata.get("feature_clip") or {"enabled": False, "bounds": {}},
        }
