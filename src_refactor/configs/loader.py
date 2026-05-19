from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(text)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}. Use .json, .yaml, or .yml.")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a mapping/object at the top level.")
    return payload


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML. Use JSON or install PyYAML.") from exc
    return yaml.safe_load(text)
