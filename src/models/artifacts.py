from __future__ import annotations

import json
import shutil
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

    def sync_from_legacy(self, spec: ModelSpec, artifact_name: str | None = None) -> ArtifactPaths:
        """Copy existing flat legacy artifacts into the new per-model folder."""
        paths = self.paths_for(spec, artifact_name)
        legacy_paths = self.legacy_paths_for(spec, artifact_name)
        self.ensure_root(paths)

        copied = self.copy_legacy_artifacts(paths, legacy_paths)
        metadata = self.build_metadata_from_legacy(spec, legacy_paths, copied)
        if metadata is not None:
            self.write_metadata(paths, metadata)

        return paths

    def copy_legacy_artifacts(self, paths: ArtifactPaths, legacy_paths: dict[str, Path]) -> dict[str, str]:
        targets = {
            "model": paths.model,
            "metrics": paths.metrics,
            "predictions": paths.predictions,
            "summary": paths.summary,
            "feature_formulas": paths.feature_formulas,
            "feature_importance": paths.feature_importance,
            "fold_feature_importance": paths.fold_feature_importance,
        }
        copied: dict[str, str] = {}
        for key, legacy_path in legacy_paths.items():
            target = targets.get(key)
            if target is None or not legacy_path.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, target)
            copied[key] = str(legacy_path)
        return copied

    def build_metadata_from_legacy(
        self,
        spec: ModelSpec,
        legacy_paths: dict[str, Path],
        copied: dict[str, str],
    ) -> ModelMetadata | None:
        legacy_meta_path = legacy_paths["metadata"]
        if not legacy_meta_path.exists():
            return None

        legacy_meta = self.read_json(legacy_meta_path)
        model_payload = self._read_torch_payload_if_available(spec, legacy_paths.get("model"))
        sequence_length = legacy_meta.get("sequence_length") or model_payload.get("sequence_length")
        model_args = legacy_meta.get("model_args") or model_payload.get("model_args") or model_payload.get("args")
        standardizer = legacy_meta.get("standardizer") or model_payload.get("standardizer")

        return ModelMetadata(
            model_key=spec.key,
            artifact_name=spec.artifact_name,
            model_type=spec.model_type,
            feature_source=spec.feature_source,
            feature_columns=list(legacy_meta.get("feature_columns") or model_payload.get("feature_columns") or []),
            symbols=list(legacy_meta.get("symbols") or model_payload.get("symbols") or []),
            label_mapping=legacy_meta.get("label_mapping") or {"short": 0, "long": 1},
            inverse_label_mapping=legacy_meta.get("inverse_label_mapping") or {"0": -1, "1": 1},
            event_filter=legacy_meta.get("event_filter") or {},
            feature_clip=legacy_meta.get("feature_clip") or {},
            train_period=legacy_meta.get("train_period") or legacy_meta.get("test_period"),
            walk_forward=self._build_walk_forward_payload(legacy_meta),
            sequence_length=int(sequence_length) if sequence_length is not None else None,
            model_args=model_args,
            standardizer=standardizer,
            legacy_artifacts=copied,
        )

    @staticmethod
    def _build_walk_forward_payload(legacy_meta: dict[str, Any]) -> dict[str, Any] | None:
        keys = {
            "n_splits": "wfv_n_splits",
            "purge_gap": "wfv_purge_gap",
            "split_mode": "wfv_split_mode",
            "monthly_train_months": "wfv_monthly_train_months",
            "monthly_test_months": "wfv_monthly_test_months",
            "monthly_window_mode": "wfv_monthly_window_mode",
        }
        payload = {
            target_key: legacy_meta[source_key]
            for target_key, source_key in keys.items()
            if source_key in legacy_meta
        }
        return payload or None

    @staticmethod
    def _read_torch_payload_if_available(spec: ModelSpec, model_path: Path | None) -> dict[str, Any]:
        if spec.model_type == "lightgbm" or model_path is None or not model_path.exists():
            return {}
        try:
            import torch
        except ImportError:
            return {}
        payload = torch.load(model_path, map_location="cpu")
        return payload if isinstance(payload, dict) else {}

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
