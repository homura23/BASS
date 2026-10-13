#!/usr/bin/env python3
"""Create a deterministic, lazy common-corruption evaluation manifest."""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freya_pro.data.manifest import derive_seed
from freya_pro.utils.config import load_config


def _label(value: object) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"ai", "fake", "synthetic", "1"}:
            return 1
        if value in {"human", "real", "bona_fide", "0"}:
            return 0
    return int(value)


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_subset(rows: list[dict], cap: int, seed: int) -> list[dict]:
    if cap <= 0:
        return rows
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["split"]), _label(row["label"])), []).append(row)
    selected: list[dict] = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda row: hashlib.sha256(
            f"{seed}|{key[0]}|{key[1]}|{row['track_id']}".encode("utf-8")
        ).digest())
        selected.extend(group[:cap])
    return sorted(selected, key=lambda row: (str(row["split"]), str(row["track_id"])))


def build_manifest(
    config_path: str, output_path: str, splits: list[str] | None = None,
    population: str = "all", assignment_mode: str | None = None,
    max_tracks_per_class_per_split: int | None = None,
    paired_real_tracks: int | None = None,
    paired_synthetic_per_generator: int | None = None,
    allow_underfilled_paired_synthetic: bool = False,
    paired_generators: list[str] | None = None,
) -> int:
    cfg = load_config(config_path)
    robustness = cfg["robustness"]
    evaluation = robustness["evaluation"]
    assignment_mode = str(
        assignment_mode or evaluation.get("assignment_mode", "all")
    )
    source_path = Path(str(evaluation["source_manifest"]))
    selected_splits = set(splits or evaluation.get("splits", ["validation", "test"]))
    operations = list(evaluation.get("operations", []))
    operation_ids = [str(item["id"]) for item in operations]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("evaluation operation ids must be unique")
    expected_count = int(evaluation.get("expected_operation_count", len(operations)))
    if len(operations) != expected_count:
        raise ValueError(
            f"Expected {expected_count} active corruption operations, got {len(operations)}"
        )
    if assignment_mode not in {"all", "one_per_track"}:
        raise ValueError(f"Unsupported assignment_mode: {assignment_mode!r}")

    sources: list[dict] = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            label = _label(source["label"])
            population_matches = (
                population == "all"
                or (population == "real" and label == 0)
                or (population == "synthetic" and label == 1)
            )
            if (str(source["split"]) in selected_splits
                    and bool(source.get("audio_exists", True))
                    and population_matches):
                sources.append(source)
    track_cap = (
        int(evaluation.get("max_tracks_per_class_per_split", 0))
        if max_tracks_per_class_per_split is None
        else int(max_tracks_per_class_per_split)
    )
    sources = _stable_subset(
        sources, track_cap, int(evaluation["mapping_seed"])
    )
    if paired_real_tracks is not None or paired_synthetic_per_generator is not None:
        if len(selected_splits) != 1 or population != "all" or assignment_mode != "all":
            raise ValueError("Paired sampling requires exactly one split, population=all, assignment_mode=all")
        real_cap = int(paired_real_tracks or 0)
        synthetic_cap = int(paired_synthetic_per_generator or 0)
        if real_cap <= 0 or synthetic_cap <= 0:
            raise ValueError("Paired real and per-generator synthetic caps must be positive")
        strata: dict[str, list[dict]] = {}
        for source in sources:
            if _label(source["label"]) == 0:
                key = "real"
            else:
                key = f'{source.get("platform", "")}:{source.get("version", "")}'
            strata.setdefault(key, []).append(source)
        selected: list[dict] = []
        available_generators = {key for key in strata if key != "real"}
        expected_generators = sorted(set(paired_generators or available_generators))
        missing_generators = set(expected_generators) - available_generators
        if missing_generators:
            raise ValueError(f"Paired generators absent from selected split: {sorted(missing_generators)}")
        for key in ["real", *expected_generators]:
            group = strata.get(key, [])
            cap = real_cap if key == "real" else synthetic_cap
            group.sort(key=lambda row: hashlib.sha256(
                f'{evaluation["mapping_seed"]}|paired|{key}|{row["track_id"]}'.encode("utf-8")
            ).digest())
            if len(group) < cap and (
                    key == "real" or not allow_underfilled_paired_synthetic
            ):
                raise ValueError(f"Paired stratum {key!r} has only {len(group)} tracks; need {cap}")
            if len(group) < cap:
                print(
                    f"Paired stratum {key!r} has only {len(group)} tracks; "
                    f"using all available tracks instead of requested {cap}.",
                    file=sys.stderr,
                )
                cap = len(group)
            selected.extend(group[:cap])
        sources = sorted(selected, key=lambda row: (_label(row["label"]), str(row["track_id"])))

    resource_cfg = robustness.get("resources", {})
    resource_roots = {"demand": resource_cfg.get("demand_dir"), "rir": resource_cfg.get("rir_dir")}
    resource_files = {
        key: sorted(Path(str(root)).rglob("*.wav")) if root else []
        for key, root in resource_roots.items()
    }
    artifact_resources = {
        "encodec": resource_cfg.get("encodec_weights"),
        "dac": resource_cfg.get("dac_weights"),
    }
    rows: list[dict] = []
    for source in sources:
        split = str(source["split"])
        base = {
            "track_id": str(source["track_id"]), "split": split,
            "audio_path": str(source["audio_path"]),
            "audio_sha256": str(source.get("audio_sha256", "")),
            "dataset": str(source.get("dataset", "")),
            "platform": str(source.get("platform", "")),
            "version": str(source.get("version", "")),
            "generator_id": (
                f'{source.get("platform", "")}:{source.get("version", "")}'
                if _label(source["label"]) == 1 else "jamendo:raw_30s"
            ),
            "label": _label(source["label"]),
        }
        if bool(evaluation.get("include_clean", True)):
            rows.append({**base, "operation_id": "clean", "operation": "clean",
                         "provenance": "control", "parameters": {},
                         "assignment_mode": assignment_mode,
                         "seed": derive_seed(int(evaluation["mapping_seed"]), base["track_id"], split, "clean")})
        selected_operations = operations
        if assignment_mode == "one_per_track":
            assignment_seed = derive_seed(
                int(evaluation["mapping_seed"]), base["track_id"], split,
                "fixed_corruption_assignment",
            )
            selected_operations = [operations[assignment_seed % len(operations)]]
        for item in selected_operations:
            operation_id = str(item["id"])
            case_seed = derive_seed(int(evaluation["mapping_seed"]), base["track_id"], split, operation_id)
            case = {**base, "operation_id": operation_id, "operation": str(item["operation"]),
                    "provenance": str(item.get("provenance", "unspecified")),
                    "parameters": dict(item.get("parameters", {})), "seed": case_seed,
                    "assignment_mode": assignment_mode}
            resource_key = item.get("resource")
            if resource_key:
                resource_name = str(resource_key)
                files = resource_files.get(resource_name, [])
                if files:
                    resource_path = files[case_seed % len(files)]
                elif artifact_resources.get(resource_name):
                    resource_path = Path(str(artifact_resources[resource_name]))
                    if not resource_path.is_file():
                        raise FileNotFoundError(resource_path)
                else:
                    raise FileNotFoundError(f"No files for resource {resource_key!r}")
                case.update(resource_id=str(resource_key), resource_path=str(resource_path),
                            resource_sha256=_sha256(resource_path))
            rows.append(case)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/robustness/robustness_config.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument(
        "--population", choices=["all", "real", "synthetic"], default="all"
    )
    parser.add_argument(
        "--assignment-mode", choices=["all", "one_per_track"], default=None
    )
    parser.add_argument("--paired-real-tracks", type=int, default=None)
    parser.add_argument(
        "--paired-synthetic-per-generator", type=int, default=None,
        help="Per-generator upper cap; with --allow-underfilled, use min(cap, available).",
    )
    parser.add_argument(
        "--paired-generator", action="append", dest="paired_generators",
        help="Restrict deterministic synthetic strata; repeat for each generator.",
    )
    parser.add_argument(
        "--allow-underfilled-paired-synthetic",
        action="store_true",
        help="Use all available tracks for synthetic strata below the requested cap",
    )
    parser.add_argument(
        "--max-tracks-per-class-per-split",
        type=int,
        default=None,
        help=(
            "Override the configured per-label, per-split track cap; "
            "use 0 to include every matching track"
        ),
    )
    args = parser.parse_args()
    count = build_manifest(
        args.config,
        args.out,
        args.splits,
        args.population,
        args.assignment_mode,
        args.max_tracks_per_class_per_split,
        args.paired_real_tracks,
        args.paired_synthetic_per_generator,
        args.allow_underfilled_paired_synthetic,
        args.paired_generators,
    )
    print(f"cases={count} out={args.out}")


if __name__ == "__main__":
    main()
