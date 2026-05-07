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

    def run_train(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("train", argv)

    def run_production(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("production", argv)

    def run_walk_forward(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("wfv", argv)

    def run_backtest(self, argv: Sequence[str] | None = None) -> None:
        self.run_legacy("backtest", argv)
