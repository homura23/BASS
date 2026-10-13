#!/usr/bin/env python3
"""Fail-closed R0 clean-sweep data and loader preflight."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freya_pro.data.dataset import make_audio_loader
from freya_pro.utils.config import load_config
from freya_pro.utils.run_io import public_path, write_json_atomic


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def identity(row: dict[str, str]) -> str:
    return row.get("track_id") or row.get("source_sha256") or row.get("full_path") or row.get("path", "")


def audit_rows(path: str, expected_labels: set[int], expected_split: str) -> tuple[list[dict[str, str]], dict]:
    rows = read_rows(path)
    labels = {int(row["label"]) for row in rows}
    if labels != expected_labels:
        raise ValueError(f"Unexpected labels in {path}: {sorted(labels)}")
    splits = {row.get("split", expected_split).lower() for row in rows}
    if splits != {expected_split}:
        raise ValueError(f"Unexpected splits in {path}: {sorted(splits)}")
    paths = [row.get("full_path") or row.get("path") for row in rows]
    missing = [value for value in paths if not value or not Path(value).is_file()]
    ids = [identity(row) for row in rows]
    if missing:
        raise FileNotFoundError(f"{path} has {len(missing)} missing audio paths; first={missing[:3]}")
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} has empty or duplicate track identities")
    return rows, {"path": public_path(path), "rows": len(rows), "labels": sorted(labels),
                  "split": expected_split, "missing_audio": 0, "unique_track_ids": len(set(ids))}


def loader_smoke(path: str, cfg: dict, train: bool) -> dict:
    smoke_cfg = dict(cfg)
    smoke_cfg["data"] = dict(cfg["data"])
    smoke_cfg["data"].update(max_batch_duration_seconds=1.0, num_workers=48,
                             auto_convert_mp3=False, batch_size=1)
    loader = make_audio_loader(path, smoke_cfg, train=train, batch_size=1,
                               num_workers=48, drop_last=False)
    batch = next(iter(loader))
    shapes = {name: list(value.shape) for name, value in batch["audio"].items()}
    if not all(torch.isfinite(value).all() for value in batch["audio"].values()):
        raise ValueError(f"Non-finite loader output: {path}")
    return {"path": public_path(path), "train": train, "branch_shapes": shapes,
            "labels": batch["label"].tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/r0.yaml")
    parser.add_argument("--out", default="runs/preflight/r0_preflight.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_rows, train_report = audit_rows(cfg["data"]["train_csv"], {0}, "train")
    val_rows, val_report = audit_rows(cfg["data"]["val_csv"], {0}, "validation")
    train_ids = {identity(row) for row in train_rows}
    val_ids = {identity(row) for row in val_rows}
    if train_ids & val_ids:
        raise ValueError("R0 has track identity overlap between train and validation")
    # Per project decision, known Jamendo source-hash duplicates are reported
    # but deliberately excluded from the pass/fail gate.
    train_hashes = {row.get("source_sha256") for row in train_rows if row.get("source_sha256")}
    validation_hashes = {
        row.get("source_sha256") for row in val_rows if row.get("source_sha256")
    }
    report = {
        "status": "pass", "config": public_path(args.config),
        "protocol": "strict_real_centered_r0_v1",
        "synthetic_visible": False,
        "datasets": {"train": train_report, "real_validation": val_report},
        "track_overlap_train_validation": 0,
        "ignored_source_hash_overlap_by_project_policy": len(
            train_hashes & validation_hashes
        ),
        "loader_smoke": {
            "train": loader_smoke(cfg["data"]["train_csv"], cfg, True),
            "real_validation": loader_smoke(cfg["data"]["val_csv"], cfg, False),
        },
    }
    destination = Path(args.out)
    write_json_atomic(destination, report)
    print(json.dumps({"status": "pass", "out": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
