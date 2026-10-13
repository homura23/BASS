"""Evaluate a CARLA base or one generator LoRA on an existing CSV split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ..data.dataset import make_audio_loader
from ..data.embedding_cache import (
    build_csv_embedding_cache, load_csv_embedding_cache, save_embedding_cache,
)
from ..models.loader import load_carla_checkpoint
from ..models.lora import load_lora_checkpoint
from ..utils.config import resolve_device
from ..utils.run_io import atomic_output_path, model_fingerprint, public_path, publicize_row, sha256_file
from ..utils.sofia_bridge import build_encoders, encoder_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument(
        "--crop-mode",
        choices=("start", "center"),
        default=None,
        help="Override the checkpoint eval crop mode for this evaluation.",
    )
    parser.add_argument("--legacy-text", action="store_true",
                        help="Historical output only; formal protocol uses JSONL")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, cfg = load_carla_checkpoint(args.base, device)
    if args.lora:
        load_lora_checkpoint(model, args.lora, base_checkpoint=args.base, strict=True)
    model.eval()
    crop_mode = args.crop_mode or str(
        cfg.get("data", {}).get("eval_crop_mode", "center")
    )
    eval_cfg = dict(cfg)
    eval_cfg["data"] = dict(cfg.get("data", {}))
    eval_cfg["data"]["eval_crop_mode"] = crop_mode
    cache_path = Path(args.embedding_cache).resolve() if args.embedding_cache else None
    if cache_path and cache_path.is_file():
        payload = load_csv_embedding_cache(
            cache_path, csv_path=args.csv, cfg=eval_cfg, crop_mode=crop_mode,
        )
    elif cache_path:
        encoders = build_encoders(cfg["audio_branches"], device, cfg.get("encoder_compatibility"))
        payload = build_csv_embedding_cache(
            encoders, args.csv, eval_cfg, device, crop_mode=crop_mode,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        save_embedding_cache(cache_path, payload)
    else:
        payload = None
        encoders = build_encoders(cfg["audio_branches"], device, cfg.get("encoder_compatibility"))
        loader = make_audio_loader(
            args.csv, eval_cfg, train=False,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    scores = []
    labels = []
    if payload is not None:
        labels = payload["labels"].tolist()
        count = len(labels)
        for start in range(0, count, 256):
            embeddings = {
                name: value[start:start + 256].to(device)
                for name, value in payload["embeddings"].items()
            }
            scores.extend(model.score(embeddings).cpu().tolist())
    else:
        for batch in loader:
            embeddings = {
                name: encoder(
                    batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
                    sample_rate=encoder.target_sr,
                ).to(device)
                for name, encoder in encoders.items()
            }
            scores.extend(model.score(embeddings).cpu().tolist())
            labels.extend(batch["label"].tolist())
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(scores):
        raise ValueError("CSV row count does not match score count")
    fingerprint = model_fingerprint(args.base, args.lora)
    provenance = {
        "base_checkpoint": public_path(args.base),
        "base_checkpoint_sha256": sha256_file(args.base),
        "lora_checkpoint": public_path(args.lora) if args.lora else None,
        "lora_checkpoint_sha256": sha256_file(args.lora) if args.lora else None,
        "model_fingerprint": fingerprint,
        "evaluation_csv": public_path(args.csv),
        "evaluation_csv_sha256": sha256_file(args.csv),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, (row, score, label) in enumerate(zip(rows, scores, labels)):
                if args.legacy_text:
                    handle.write(f"{index} {score:.8f} {'fake' if int(label) else 'real'}\n")
                    continue
                value = {
                    **publicize_row(row), **provenance, "label": int(label), "score": float(score),
                    "operation_id": row.get("operation_id") or "clean",
                    "generator_id": row.get("source") or row.get("generator") or "",
                    "status": "ok",
                }
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"samples={len(scores)} out={path}")


if __name__ == "__main__":
    main()
