"""Fail-closed helpers for the development-frozen strict few-shot protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.run_io import sha256_file

PROTOCOL = "strict_fewshot_development_frozen_v1"
NORMALIZATION = "support_real_mean_all_std_zscore"


def load_frozen_recipe(path: str | Path) -> dict[str, Any]:
    recipe_path = Path(path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("protocol") != PROTOCOL or recipe.get("status") != "frozen":
        raise ValueError(f"Not a frozen {PROTOCOL} recipe: {recipe_path}")
    development = set(map(str, recipe.get("development_generators", [])))
    targets = set(map(str, recipe.get("target_generators", [])))
    if not development or not targets or development & targets:
        raise ValueError("Recipe requires non-empty, disjoint development and target generators")
    if recipe.get("score_normalization") != NORMALIZATION:
        raise ValueError("Strict recipe must use support-only score normalization")
    threshold = recipe.get("normalized_threshold")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Frozen recipe has no numeric normalized_threshold")
    if not isinstance(recipe.get("shot_recipes"), dict):
        raise ValueError("Frozen recipe has no shot_recipes")
    augmentation = str(recipe.get("support_augmentation", "clean"))
    if augmentation not in {"clean", "randaugment"}:
        raise ValueError("Unsupported frozen support_augmentation")
    if augmentation == "randaugment":
        policy = recipe.get("randaugment", {})
        if int(policy.get("num_layers", 0)) <= 0 or not 0.0 <= float(policy.get("apply_probability", -1.0)) <= 1.0:
            raise ValueError("Frozen RandAugment policy is invalid")
    return recipe


def validate_target_run(
    recipe: dict[str, Any], *, generator: str, arm: str, shots: int,
    rank: int, alpha: float, targets: list[str], lr: float,
    weight_decay: float, steps: int, support_rows: list[dict[str, str]],
    support_augmentation: str = "clean",
    randaugment_num_layers: int = 1,
    randaugment_apply_probability: float = 0.9,
) -> None:
    if generator not in set(map(str, recipe["target_generators"])):
        raise ValueError(f"Generator {generator!r} is not a formal target in the frozen recipe")
    if arm not in set(map(str, recipe.get("arms", []))):
        raise ValueError(f"Arm {arm!r} is not registered in the frozen recipe")
    shot_recipe = recipe["shot_recipes"].get(str(shots))
    if not isinstance(shot_recipe, dict):
        raise ValueError(f"Shot size N={shots} is not registered in the frozen recipe")
    lora = recipe.get("lora", {})
    expected = {
        "rank": int(lora["rank"]), "alpha": float(lora["alpha"]),
        "targets": list(lora["targets"]), "lr": float(lora["lr"]),
        "weight_decay": float(lora["weight_decay"]),
        "steps": int(shot_recipe["steps"]),
        "support_augmentation": str(recipe.get("support_augmentation", "clean")),
    }
    actual = {
        "rank": int(rank), "alpha": float(alpha), "targets": list(targets),
        "lr": float(lr), "weight_decay": float(weight_decay), "steps": int(steps),
        "support_augmentation": str(support_augmentation),
    }
    if expected["support_augmentation"] == "randaugment":
        policy = recipe.get("randaugment", {})
        expected.update({
            "randaugment_num_layers": int(policy["num_layers"]),
            "randaugment_apply_probability": float(policy["apply_probability"]),
        })
        actual.update({
            "randaugment_num_layers": int(randaugment_num_layers),
            "randaugment_apply_probability": float(randaugment_apply_probability),
        })
    if actual != expected:
        raise ValueError(f"Run does not match frozen adaptation recipe: expected={expected} actual={actual}")
    real = sum(int(row["label"]) == 0 for row in support_rows)
    fake = sum(int(row["label"]) == 1 for row in support_rows)
    exception = recipe.get("underfilled_support", {}).get(generator)
    if exception is None:
        if (real, fake) != (shots, shots):
            raise ValueError(f"Strict N-shot support must contain exactly N real + N fake: {(real, fake)}")
    else:
        expected_real = int(exception.get("real", shots))
        expected_fake = int(exception.get("fake", shots))
        nominal_shots = int(exception.get("nominal_shots", shots))
        if nominal_shots != shots or real != expected_real or fake != expected_fake:
            raise ValueError(
                f"Underfilled support mismatch for {generator}: expected real/fake="
                f"{expected_real}/{expected_fake}, nominal N={shots}; got {real}/{fake}"
            )
        if not (0 < expected_fake < shots and expected_real == shots):
            raise ValueError(f"Invalid underfilled support exception for {generator}: {exception}")


def recipe_identity(path: str | Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
