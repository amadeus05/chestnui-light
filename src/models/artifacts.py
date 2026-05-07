from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import config as cfg
from src.models.contracts import ArtifactPaths, ModelMetadata, ModelSpec


class ArtifactStore:
    """Single place for model artifact paths and JSON persistence."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or cfg.MODELS_DIR)

    def paths_for(self, spec: ModelSpec, artifact_name: str | None = None) -> ArtifactPaths:
        name = artifact_name or spec.artifact_name
        root = self.models_dir / name
        suffix = ".joblib" if spec.model_type == "lightgbm" else ".pt"
        return ArtifactPaths(
            root=root,
            model=root / f"model{suffix}",
            metadata=root / "metadata.json",
            metrics=root / "metrics.json",
            predictions=root / "oos_predictions.csv",
            summary=root / "summary.json",
            feature_formulas=root / "feature_formulas.json",
            feature_importance=root / "feature_importance.csv" if spec.model_type == "lightgbm" else None,
            fold_feature_importance=root / "fold_feature_importance.csv" if spec.model_type == "lightgbm" else None,
        )

    def legacy_paths_for(self, spec: ModelSpec, artifact_name: str | None = None) -> dict[str, Path]:
        """Map current flat legacy filenames while migration is in progress."""
        name = artifact_name or spec.artifact_name
        model_suffix = ".joblib" if spec.model_type == "lightgbm" else ".pt"
        return {
            "model": self.models_dir / f"{name}{model_suffix}",
            "metadata": self.models_dir / f"{name}_features.json",
            "metrics": self.models_dir / f"{name}_metrics.json",
            "predictions": self.models_dir / f"{spec.default_predictions_name}.csv",
            "summary": self.models_dir / f"{spec.default_predictions_name}_summary.json",
            "feature_formulas": self.models_dir / f"{name}_feature_formulas.json",
            "feature_importance": self.models_dir / f"{name}_feature_importance.csv",
            "fold_feature_importance": self.models_dir / f"{name}_fold_feature_importance.csv",
        }

    def ensure_root(self, paths: ArtifactPaths) -> None:
        paths.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._json_payload(payload), indent=2), encoding="utf-8")

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_metadata(self, paths: ArtifactPaths, metadata: ModelMetadata) -> None:
        self.write_json(paths.metadata, metadata.to_dict())

    @staticmethod
    def _json_payload(payload: Any) -> Any:
        if isinstance(payload, ModelMetadata):
            return payload.to_dict()
        if is_dataclass(payload):
            return asdict(payload)
        return payload
