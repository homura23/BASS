"""Invocation checks for public R0/R1 support-only LoRA training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .r1_only import ALLOWED_ARMS, sha256_file, validate_r1_only_plan


def validate_training_invocation(
    plan_path: Path,
    cell_id: str,
    variant: str,
    *,
    base: Path,
    generator: str,
    rank: int,
    alpha: float,
    targets: list[str],
    lr: float,
    weight_decay: float,
    steps: int,
) -> dict[str, Any]:
    import json
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_r1_only_plan(plan)
    if cell_id not in {"r0_lora", "r1_lora", "r1_only"}:
        raise ValueError(f"Unknown public CARLA ablation cell: {cell_id}")
    if variant not in {"r0", "r1", "experts", "projectors", "experts+projectors"}:
        raise ValueError(f"Unsupported public CARLA variant: {variant}")
    if rank <= 0 or steps <= 0 or not targets:
        raise ValueError("LoRA rank, steps, and targets must be positive")
    return {
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "cell_id": cell_id,
        "variant": variant,
        "base_arm": "r1" if "r1" in cell_id or variant == "r1" else "r0",
        "base_checkpoint_sha256": sha256_file(base),
        "generator": generator,
        "rank": rank,
        "alpha": alpha,
        "targets": list(targets),
        "lr": lr,
        "weight_decay": weight_decay,
        "steps": steps,
    }
