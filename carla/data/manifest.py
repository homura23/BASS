"""Track manifests, deterministic splits, registries and audio materialization."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
try:
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover - manifest split tests need no audio backend
    torchaudio = None

from .corruptions import ensure_stereo
from .audio_io import load_audio_cached
from .qc import quality_gate


def _require_torchaudio():
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to inspect or materialize audio")
    return torchaudio


@dataclass
class TrackRecord:
    track_id: str
    full_path: str
    label: int
    source: str = ""
    generator_name: str = ""
    generator_version: str = ""
    artist_group_id: str = ""
    album_group_id: str = ""
    song_group_id: str = ""
    split: str = ""
    native_sample_rate: int = 0
    native_channels: int = 0
    source_sha256: str = ""


@dataclass
class VariantRecord:
    variant_id: str
    track_id: str
    condition_id: str
    replicate_id: int
    relative_audio_path: str
    output_sha256: str
    output_sample_rate: int
    output_channels: int
    output_duration_samples: int
    parent_variant_id: str
    operation_chain_id: str
    random_seed: int
    origin_label: int
    split: str
    accepted: bool
    rejection_reason: str = ""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_track_id(path: str | Path, source_hash: Optional[str] = None) -> str:
    value = source_hash or sha256_file(path)
    # Base32 without padding gives a filename-safe, label-neutral identifier.
    import base64

    encoded = base64.b32encode(bytes.fromhex(value)).decode("ascii").lower().rstrip("=")
    return f"trk_{encoded[:12]}"


def derive_seed(master_seed: int, *parts: object) -> int:
    payload = "::".join([str(master_seed), *(str(part) for part in parts)])
    value = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big", signed=False)


def read_track_csv(path: str | Path) -> list[TrackRecord]:
    """Read CARLA-style CSVs and preserve optional grouping columns."""

    records: list[TrackRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        path_key = "full_path" if "full_path" in reader.fieldnames else "path"
        if path_key not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain full_path/path and label columns")
        for row in reader:
            full_path = str(row[path_key]).strip()
            if not full_path:
                raise ValueError("CSV contains an empty audio path")
            source_hash = row.get("source_sha256", "")
            if not source_hash:
                source_hash = sha256_file(full_path)
            track_id = row.get("track_id") or stable_track_id(full_path, source_hash)
            records.append(
                TrackRecord(
                    track_id=track_id,
                    full_path=full_path,
                    label=int(row["label"]),
                    source=row.get("source", ""),
                    generator_name=row.get("generator_name", row.get("generator", "")),
                    generator_version=row.get("generator_version", ""),
                    artist_group_id=row.get("artist_group_id", row.get("artist", "")),
                    album_group_id=row.get("album_group_id", row.get("album", "")),
                    song_group_id=row.get("song_group_id", row.get("song", "")),
                    split=row.get("split", ""),
                    source_sha256=source_hash,
                )
            )
    return records


def inspect_track(record: TrackRecord) -> TrackRecord:
    audio = _require_torchaudio()
    waveform, sample_rate = audio.load(record.full_path)
    record.native_sample_rate = int(sample_rate)
    record.native_channels = int(waveform.shape[0])
    return record


def canonical_load(
    path: str | Path,
    *,
    auto_convert_mp3: bool = True,
    audio_cache_dir: str | Path | None = None,
) -> tuple[torch.Tensor, int]:
    """Decode once, preserve native rate, and return float32 stereo PCM."""

    waveform, sample_rate = load_audio_cached(
        path,
        target_sr=None,
        mono=False,
        normalize=False,
        strict=True,
        auto_convert_mp3=auto_convert_mp3,
        cache_dir=audio_cache_dir,
    )
    return ensure_stereo(waveform), int(sample_rate)


def canonical_save(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> str:
    audio = _require_torchaudio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = ensure_stereo(waveform).detach().cpu().to(torch.float32)
    audio.save(str(path), waveform, int(sample_rate), encoding="PCM_F")
    return sha256_file(path)


def _group_key(record: TrackRecord) -> str:
    return (
        record.song_group_id
        or record.artist_group_id
        or record.album_group_id
        or record.track_id
        or record.source_sha256
    )


def assign_splits(
    records: Sequence[TrackRecord],
    *,
    master_seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> list[TrackRecord]:
    """Assign deterministic group splits while approximately stratifying labels."""

    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave a test split")
    if any(record.split for record in records) and not all(record.split for record in records):
        raise ValueError("Either every input row must provide split or none may do so")
    if all(record.split for record in records):
        return [record for record in records]

    groups: dict[str, list[TrackRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)
    group_items = list(groups.items())
    random.Random(master_seed).shuffle(group_items)

    target_counts = {
        "train": round(len(records) * train_ratio),
        "validation": round(len(records) * validation_ratio),
    }
    target_counts["test"] = len(records) - target_counts["train"] - target_counts["validation"]
    counts = {"train": 0, "validation": 0, "test": 0}
    for _, group in group_items:
        candidates = sorted(
            counts,
            key=lambda split: (
                counts[split] >= target_counts[split],
                counts[split] / max(target_counts[split], 1),
                split,
            ),
        )
        split = candidates[0]
        for record in group:
            record.split = split
        counts[split] += len(group)
    return [record for record in records]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any] | Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_variant_id(track_id: str, condition_id: str, replicate_id: int) -> str:
    digest = hashlib.sha1(
        f"{track_id}::{condition_id}::{replicate_id}".encode("utf-8")
    ).hexdigest()[:10]
    return f"var_{digest}"


def materialize_variant(
    *,
    record: TrackRecord,
    waveform: torch.Tensor,
    sample_rate: int,
    condition_id: str,
    parameters: Mapping[str, Any],
    operation: str,
    output_root: str | Path,
    replicate_id: int,
    seed: int,
    parent_variant_id: str = "clean",
    operation_chain_id: Optional[str] = None,
    audio_cache_dir: str | Path | None = None,
) -> tuple[VariantRecord, dict[str, Any]]:
    """Run QC, save a variant and return both records."""

    output_root = Path(output_root)
    relative = Path("audio") / record.split / record.track_id / (
        f"{record.track_id}__{condition_id}__r{replicate_id:02d}.wav"
    )
    destination = output_root / relative
    clean, native_sr = canonical_load(
        record.full_path,
        auto_convert_mp3=True,
        audio_cache_dir=audio_cache_dir,
    )
    qc = quality_gate(
        clean,
        waveform,
        sample_rate=sample_rate,
        operation=operation,
        parameters=parameters,
    )
    accepted = bool(qc["accepted"])
    output_hash = ""
    if accepted:
        output_hash = canonical_save(destination, waveform, sample_rate)
    variant = VariantRecord(
        variant_id=build_variant_id(record.track_id, condition_id, replicate_id),
        track_id=record.track_id,
        condition_id=condition_id,
        replicate_id=int(replicate_id),
        relative_audio_path=str(relative),
        output_sha256=output_hash,
        output_sample_rate=int(sample_rate),
        output_channels=int(waveform.shape[0]),
        output_duration_samples=int(waveform.shape[-1]),
        parent_variant_id=parent_variant_id,
        operation_chain_id=operation_chain_id or condition_id,
        random_seed=int(seed),
        origin_label=int(record.label),
        split=record.split,
        accepted=accepted,
        rejection_reason=str(qc.get("rejection_reason", "")),
    )
    return variant, qc


def audit_split_leakage(records: Iterable[TrackRecord], variants: Iterable[VariantRecord]) -> list[str]:
    """Return human-readable leakage errors; an empty list means pass."""

    errors: list[str] = []
    track_splits: dict[str, str] = {}
    hash_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    for record in records:
        if record.track_id in track_splits and track_splits[record.track_id] != record.split:
            errors.append(f"track_id crosses split: {record.track_id}")
        track_splits[record.track_id] = record.split
        if record.source_sha256:
            if (
                record.source_sha256 in hash_splits
                and hash_splits[record.source_sha256] != record.split
            ):
                errors.append(f"source hash crosses split: {record.source_sha256}")
            hash_splits[record.source_sha256] = record.split
        group = _group_key(record)
        if group in group_splits and group_splits[group] != record.split:
            errors.append(f"group crosses split: {group}")
        group_splits[group] = record.split
    for variant in variants:
        if variant.track_id not in track_splits:
            errors.append(f"variant references unknown track: {variant.track_id}")
        elif track_splits[variant.track_id] != variant.split:
            errors.append(f"variant crosses split: {variant.variant_id}")
    return errors
