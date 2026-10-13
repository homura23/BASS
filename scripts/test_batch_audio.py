#!/usr/bin/env python3
"""Run CARLA inference on a CSV manifest.

This is the public CARLA entry point corresponding to SOFIA's batch testing
workflow. It uses the same CSV contract (full_path,label plus optional
metadata), while the score is produced by the CARLA R0/R1 density model.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CARLA R0/R1 checkpoint")
    parser.add_argument("--csv", required=True, help="CSV with full_path,label columns")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--lora", default=None, help="Optional support-only LoRA checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--crop-mode", choices=("start", "center"), default=None)
    parser.add_argument("--legacy-text", action="store_true")
    parser.add_argument("--sofia-root", default=None)
    args = parser.parse_args()
    if args.sofia_root:
        os.environ["SOFIA_ROOT"] = str(Path(args.sofia_root).expanduser().resolve())

    # Keep the implementation in the tested CARLA evaluator so this wrapper
    # and the existing training/evaluation entry point cannot drift apart.
    import carla.eval.evaluate as evaluator

    forwarded = [
        "--base", args.checkpoint,
        "--csv", args.csv,
        "--out", args.output,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    for flag, value in (
        ("--lora", args.lora),
        ("--embedding-cache", args.embedding_cache),
        ("--crop-mode", args.crop_mode),
        ("--device", args.device),
    ):
        if value is not None:
            forwarded.extend((flag, value))
    if args.legacy_text:
        forwarded.append("--legacy-text")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *forwarded]
        evaluator.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
