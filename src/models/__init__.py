"""Unified model management layer.

Legacy training/backtest scripts stay in place. New model runners live here and
gradually absorb that logic behind a common registry and artifact contract.
"""

from src.models.registry import get_model_spec, list_model_specs

__all__ = ["get_model_spec", "list_model_specs"]
