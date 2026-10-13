"""Build frozen SOFIA clean/corrupt embedding caches from variant manifests."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
import torchaudio

from .audio_io import load_audio_cached
from .corruptions import ensure_stereo
from .manifest import read_jsonl
from ..utils.sofia_bridge import build_encoders, encoder_device
from ..utils.run_io import torch_save_atomic


def _signature(*values: object) -> str:
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_rows_from_variants(
    variant_manifest: str | Path,
    audio_root: str | Path,
    *,
    condition_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Group clean and accepted variants into paired cache rows."""

    rows = read_jsonl(variant_manifest)
    root = Path(audio_root)
    allowed = set(condition_ids or [])
    clean_by_track: dict[str, dict[str, Any]] = {}
    variants: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("accepted", True):
            continue
        if row.get("condition_id") == "clean":
            clean_by_track[row["track_id"]] = row
        elif not allowed or row.get("condition_id") in allowed:
            variants.append(row)
    output = []
    for variant in variants:
        clean = clean_by_track.get(variant["track_id"])
        if clean is None:
            raise ValueError(f"Missing clean variant for {variant['track_id']}")
        output.append(
            {
                "track_id": variant["track_id"],
                "condition_id": variant["condition_id"],
                "split": variant.get("split", ""),
                "origin_label": int(variant.get("origin_label", 0)),
                "clean_path": str(root / clean["relative_audio_path"]),
                "corrupt_path": str(root / variant["relative_audio_path"]),
                "seed": int(variant.get("random_seed", 0)),
                "variant_id": variant.get("variant_id", ""),
            }
        )
    return output


def _encode_one(
    encoders: Mapping[str, torch.nn.Module],
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, encoder in encoders.items():
        branch_device = encoder_device(encoder)
        branch_audio = waveform.to(branch_device)
        with torch.no_grad():
            value = encoder(
                branch_audio,
                sample_rate=int(sample_rate),
            ).to(device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        output[name] = value.detach().cpu()
    return output


def _prepare_model_waveform(
    path: str,
    *,
    base_sample_rate: int,
    max_batch_duration_seconds: float,
    crop_mode: str,
    seed: int,
    audio_cache_dir: str | None,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = load_audio_cached(
        path,
        target_sr=None,
        mono=False,
        normalize=False,
        strict=True,
        auto_convert_mp3=True,
        cache_dir=audio_cache_dir,
    )
    waveform = ensure_stereo(waveform)
    if int(sample_rate) != int(base_sample_rate):
        waveform = torchaudio.functional.resample(
            waveform, int(sample_rate), int(base_sample_rate)
        )
        sample_rate = int(base_sample_rate)
    target_length = (
        round(float(max_batch_duration_seconds) * int(sample_rate))
        if max_batch_duration_seconds > 0
        else 0
    )
    if target_length <= 0:
        return waveform, int(sample_rate)
    current = waveform.shape[-1]
    if current > target_length:
        if crop_mode == "start":
            start = 0
        elif crop_mode == "center":
            start = (current - target_length) // 2
        elif crop_mode == "random":
            start = random.Random(int(seed)).randint(0, current - target_length)
        else:
            raise ValueError("crop_mode must be random, start, or center")
        waveform = waveform[..., start : start + target_length]
    elif current < target_length:
        total = target_length - current
        waveform = F.pad(waveform, (total // 2, total - total // 2))
    return waveform, int(sample_rate)


def build_embedding_cache(
    rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    encoders = build_encoders(dict(cfg["audio_branches"]), device, cfg.get("encoder_compatibility"))
    data_cfg = cfg.get("data", {})
    audio_cache_dir = data_cfg.get("audio_cache_dir")
    base_sample_rate = int(data_cfg.get("base_sample_rate", 44100))
    max_batch_duration_seconds = float(
        data_cfg.get("max_batch_duration_seconds", 60.0)
    )
    crop_mode = str(data_cfg.get("crop_mode", "random"))
    clean_values: dict[str, list[torch.Tensor]] = {name: [] for name in encoders}
    corrupt_values: dict[str, list[torch.Tensor]] = {name: [] for name in encoders}
    metadata: list[dict[str, Any]] = []
    rows = list(rows)
    for row in rows:
        clean_waveform, clean_sr = _prepare_model_waveform(
            str(row["clean_path"]),
            base_sample_rate=base_sample_rate,
            max_batch_duration_seconds=max_batch_duration_seconds,
            crop_mode=crop_mode,
            seed=int(row.get("seed", 0)),
            audio_cache_dir=audio_cache_dir,
        )
        corrupt_waveform, corrupt_sr = _prepare_model_waveform(
            str(row["corrupt_path"]),
            base_sample_rate=base_sample_rate,
            max_batch_duration_seconds=max_batch_duration_seconds,
            crop_mode=crop_mode,
            seed=int(row.get("seed", 0)),
            audio_cache_dir=audio_cache_dir,
        )
        clean = _encode_one(
            encoders,
            clean_waveform,
            clean_sr,
            device=device,
        )
        corrupt = _encode_one(
            encoders,
            corrupt_waveform,
            corrupt_sr,
            device=device,
        )
        if set(clean) != set(corrupt):
            raise ValueError("Clean and corrupt encoder branches differ")
        for name in encoders:
            clean_values[name].append(clean[name].squeeze(0))
            corrupt_values[name].append(corrupt[name].squeeze(0))
        metadata.append(dict(row))

    cache_signature = _signature(
        cfg.get("audio_branches", {}),
        data_cfg.get("max_batch_duration_seconds", 0),
        data_cfg.get("crop_mode", "random"),
        [
            {
                "track_id": row.get("track_id"),
                "condition_id": row.get("condition_id"),
                "clean_path": row.get("clean_path"),
                "corrupt_path": row.get("corrupt_path"),
            }
            for row in rows
        ],
    )
    return {
        "project": "CARLA",
        "cache_version": 2,
        "clean": {name: torch.stack(values) for name, values in clean_values.items()},
        "corrupt": {
            name: torch.stack(values) for name, values in corrupt_values.items()
        },
        "metadata": metadata,
        "cache_signature": {
            "sha256": cache_signature,
            "encoder_config": cfg.get("audio_branches", {}),
        },
    }


def main() -> None:
    import argparse

    from ..utils.config import load_config, resolve_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--condition", action="append", dest="conditions", default=None)
    parser.add_argument("--audio-cache-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.audio_cache_dir:
        cfg["data"] = dict(cfg.get("data", {}))
        cfg["data"]["audio_cache_dir"] = args.audio_cache_dir
    rows = pair_rows_from_variants(
        args.manifest,
        args.audio_root,
        condition_ids=args.conditions,
    )
    cache = build_embedding_cache(rows, cfg, device=resolve_device(args.device))
    destination = Path(args.out)
    torch_save_atomic(destination, cache)
    print(f"saved={destination} rows={len(rows)} signature={cache['cache_signature']['sha256']}")


if __name__ == "__main__":
    main()
