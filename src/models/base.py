from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.models.artifacts import ArtifactStore
from src.models.contracts import ModelSpec, RunMode
from src.models.legacy import LegacyDelegatingRunner


@dataclass(slots=True)
class BaseModelRunner:
    spec: ModelSpec
    artifact_store: ArtifactStore

    def run_legacy(self, mode: RunMode, argv: Sequence[str] | None = None) -> None:
        LegacyDelegatingRunner(self.spec).run(mode, argv)

    def sync_artifacts(self) -> None:
        paths = self.artifact_store.sync_from_legacy(self.spec)
        print(f"Synced '{self.spec.key}' artifacts to {paths.root}")

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("train", argv)
        self.sync_artifacts()

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("production", argv)
        self.sync_artifacts()

    def run_walk_forward(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("wfv", argv)
        self.sync_artifacts()

    def run_backtest(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("backtest", argv)
