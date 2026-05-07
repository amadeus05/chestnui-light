from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Sequence

from src.models.contracts import ModelSpec, RunMode


@dataclass(slots=True)
class LegacyDelegatingRunner:
    """Temporary bridge from the new registry to existing legacy entrypoints."""

    spec: ModelSpec

    def run(self, mode: RunMode, argv: Sequence[str] | None = None) -> None:
        module_name = self._module_for(mode)
        if module_name is None:
            raise ValueError(f"Model '{self.spec.key}' does not support mode '{mode}'.")
        self._run_module_main(module_name, argv or [])

    def _module_for(self, mode: RunMode) -> str | None:
        if mode == "train":
            return self.spec.legacy_train_module or self.spec.legacy_production_module
        if mode == "production":
            return self.spec.legacy_production_module or self.spec.legacy_train_module
        if mode == "wfv":
            return self.spec.legacy_wfv_module
        if mode == "backtest":
            return self.spec.legacy_backtest_module
        raise ValueError(f"Unsupported run mode: {mode}")

    @staticmethod
    def _run_module_main(module_name: str, argv: Sequence[str]) -> None:
        module = importlib.import_module(module_name)
        main = getattr(module, "main", None)
        if main is None:
            raise AttributeError(f"Module '{module_name}' does not expose main().")

        old_argv = sys.argv[:]
        sys.argv = [module_name, *argv]
        try:
            main()
        finally:
            sys.argv = old_argv
