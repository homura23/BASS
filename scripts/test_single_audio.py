#!/usr/bin/env python3
"""Run CARLA inference for one audio file using the external SOFIA encoders.

The temporary one-row manifest is created outside the repository and removed
automatically. The command prints one JSON object and does not write results
to the project tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CARLA R0/R1 checkpoint")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional score threshold; score < threshold is reported as fake",
    )
    parser.add_argument(
        "--sofia-root",
        default=None,
        help="SOFIA checkout; otherwise SOFIA_ROOT or the sibling ../sofia is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio = Path(args.audio).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if args.sofia_root:
        os.environ["SOFIA_ROOT"] = str(Path(args.sofia_root).expanduser().resolve())

    import torch

    from carla.data.dataset import make_audio_loader
    from carla.models.loader import load_carla_checkpoint
    from carla.utils.config import resolve_device
    from carla.utils.run_io import public_path, sha256_file
    from carla.utils.sofia_bridge import build_encoders, encoder_device

    device = resolve_device(args.device)
    model, cfg = load_carla_checkpoint(str(checkpoint), device)
    model.eval()

    # Keep the manifest ephemeral: it is only an adapter to SOFIA's official
    # CSV-based AudioSampleDataset interface.
    with tempfile.TemporaryDirectory(prefix="carla_single_audio_") as directory:
        manifest = Path(directory) / "single.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("full_path", "label", "source"))
            writer.writeheader()
            writer.writerow({"full_path": str(audio), "label": 0, "source": "single_audio"})

        loader = make_audio_loader(
            str(manifest), cfg, train=False, batch_size=1, num_workers=0, drop_last=False,
        )
        encoders = build_encoders(
            cfg["audio_branches"], device, cfg.get("encoder_compatibility"),
        )
        with torch.inference_mode():
            batch = next(iter(loader))
            embeddings = {
                name: encoder(
                    batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
                    sample_rate=encoder.target_sr,
                ).to(device)
                for name, encoder in encoders.items()
            }
            score = float(model.score(embeddings).cpu().item())

    result = {
        "status": "ok",
        "audio": public_path(audio),
        "checkpoint": public_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_version": str(cfg.get("model_version", "unknown")),
        "score": score,
        "device": str(device),
    }
    if args.threshold is not None:
        result["threshold"] = float(args.threshold)
        result["predicted_label"] = int(score < args.threshold)
        result["predicted_class"] = "fake" if score < args.threshold else "real"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
