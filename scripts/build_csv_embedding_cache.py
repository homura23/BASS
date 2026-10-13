#!/usr/bin/env python3
"""Build or validate a frozen SOFIA branch-embedding cache for one CSV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freya_pro.data.embedding_cache import (
    build_csv_embedding_cache, save_embedding_cache,
    validate_csv_embedding_cache_metadata,
)
from freya_pro.utils.config import load_config, resolve_device
from freya_pro.utils.sofia_bridge import build_encoders


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML configuration used by the model")
    source.add_argument("--checkpoint", help="FREYA-PRO checkpoint whose embedded config is authoritative")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop-mode", choices=["start", "center"], required=True)
    parser.add_argument("--max-batch-duration-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            raise ValueError(f"Checkpoint has no valid embedded config: {args.checkpoint}")
    else:
        cfg = load_config(args.config)
    cfg["data"] = dict(cfg["data"])
    # The feature cache is the durable acceleration artifact. Direct MP3 decode
    # avoids additionally materializing every full song as a very large PCM WAV.
    cfg["data"]["auto_convert_mp3"] = False
    cfg["data"]["audio_cache_dir"] = None
    if args.max_batch_duration_seconds is not None:
        cfg["data"]["max_batch_duration_seconds"] = args.max_batch_duration_seconds
    cfg["data"]["eval_crop_mode"] = args.crop_mode
    destination = Path(args.out).resolve()
    if destination.is_file():
        metadata = validate_csv_embedding_cache_metadata(
            destination, csv_path=args.csv, cfg=cfg, crop_mode=args.crop_mode,
        )
        print(json.dumps({"status": "hit", "out": str(destination),
                          "rows": int(metadata["spec"]["rows"]),
                          "signature": metadata["signature_sha256"]}))
        return 0
    device = resolve_device(args.device or cfg.get("device"))
    encoders = build_encoders(cfg["audio_branches"], device, cfg.get("encoder_compatibility"))
    payload = build_csv_embedding_cache(
        encoders, args.csv, cfg, device, crop_mode=args.crop_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    save_embedding_cache(destination, payload)
    print(json.dumps({"status": "built", "out": str(destination),
                      "rows": int(payload["labels"].shape[0]),
                      "signature": payload["signature_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
