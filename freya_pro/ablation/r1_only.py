"""Small public protocol helpers for R0/R1 LoRA experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ALLOWED_ARMS = {"r0", "r1"}


def build_r1_only_plan() -> dict[str, Any]:
    return {
        "protocol": "carla_r0_r1_lora_v1",
        "allowed_arms": sorted(ALLOWED_ARMS),
        "target_validation_used_for_selection": False,
        "target_test_used_for_selection": False,
    }


def validate_r1_only_plan(plan: dict[str, Any]) -> None:
    if set(plan.get("allowed_arms", ())) != ALLOWED_ARMS:
        raise ValueError("CARLA public LoRA protocol supports only R0 and R1")


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
