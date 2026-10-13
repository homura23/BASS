from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    extends = value.pop("extends", None)
    if extends:
        parent = load_config(str((config_path.parent / str(extends)).resolve()))
        value = _deep_merge(parent, value)
    return value


def resolve_device(value: str | None) -> torch.device:
    if value and value != "auto":
        return torch.device(value)
    # Keep automatic CUDA unindexed: encoder construction may distribute
    # branches over all visible cards.  An explicit cuda:N means strict
    # single-card placement for the whole task.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
