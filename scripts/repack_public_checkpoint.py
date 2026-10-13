#!/usr/bin/env python3
"""Repackage one internal v1 checkpoint as a portable CARLA release artifact.

The source checkpoint is never copied verbatim: optimizer state, experiment
metadata, private CSV/cache paths, and the FREYA-PRO project identifier are
removed. Only the R0/R1 fusion and flow weights plus their portable inference
configuration are retained.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from freya_pro.utils.config import load_config  # noqa: E402
from freya_pro.utils.run_io import sha256_file, torch_save_atomic  # noqa: E402


def portable_config(source: dict, public: dict, version: str) -> dict:
    config = copy.deepcopy(public)
    source_data = source.get("data", {})
    data = config.setdefault("data", {})
    for key in (
        "base_sample_rate", "segment_seconds", "max_batch_duration_seconds",
        "normalize", "mono", "random_crop", "crop_mode", "eval_crop_mode",
        "auto_convert_mp3", "batch_size", "num_workers",
    ):
        if key in source_data:
            data[key] = source_data[key]
    data.update({
        "train_csv": "data/examples/train.csv",
        "val_csv": "data/examples/validation.csv",
        "selection_csv": None,
        "train_embedding_cache": None,
        "val_embedding_cache": None,
        "selection_embedding_cache": None,
        "audio_cache_dir": None,
    })
    config["model_version"] = version
    config["encoder_compatibility"] = {
        "mode": "checkpoint_repaired",
        "initialization_seed": int(
            source.get("encoder_compatibility", {}).get("initialization_seed", 42)
        ),
        "sofia_root": None,
    }
    # These modules define the learned input/output contract and therefore
    # must match the state dict exactly. All paths remain from the public
    # config and are relative to SOFIA_ROOT.
    config["fusion"] = copy.deepcopy(source["fusion"])
    config["flow"] = copy.deepcopy(source["flow"])
    return config


def assert_portable(value: object, location: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_portable(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_portable(child, f"{location}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/"):
            raise ValueError(f"Non-portable path in {location}: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--variant", choices=("r0", "r1"), required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_path = Path(args.source).expanduser().resolve()
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = PROJECT_ROOT / "config" / f"{args.variant}.yaml"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("arch", "v1") != "v1":
        raise ValueError("Only v1 checkpoints can be publicized")
    if "fusion" not in source or "flow" not in source:
        raise ValueError("Source checkpoint must contain fusion and flow state dicts")
    public = portable_config(source.get("config", {}), load_config(str(config_path)), args.variant)
    assert_portable(public)
    artifact = {
        "project": "CARLA",
        "arch": "v1",
        "model_version": args.variant,
        "robust_variant": args.variant,
        "fusion": source["fusion"],
        "flow": source["flow"],
        "config": public,
        "stage": f"final_{args.variant}",
        "epoch": int(source.get("epoch", 0)),
        "metadata": {
            "publicized": True,
            "feature_contract": "checkpoint_repaired",
            "source_checkpoint_sha256": sha256_file(source_path),
            "source_stage": str(source.get("stage", "unknown")),
            "source_epoch": int(source.get("epoch", 0)),
        },
        "optimizer": None,
        "training_state": {},
    }
    torch_save_atomic(args.out, artifact)
    print(f"wrote={Path(args.out).resolve()} bytes={Path(args.out).stat().st_size}")


if __name__ == "__main__":
    main()
