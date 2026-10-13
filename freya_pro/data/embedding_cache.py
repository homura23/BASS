"""Validated disk caches for frozen SOFIA branch embeddings."""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import configure_data_worker_threads, make_audio_loader
from .qc import validate_embedding_tensor
from ..utils.run_io import public_path, publicize_row, sha256_file, torch_save_atomic, write_json_atomic
from ..utils.sofia_bridge import encoder_device


CACHE_VERSION = 1
PREPROCESS_KEYS = (
    "base_sample_rate", "segment_seconds", "max_batch_duration_seconds",
    "normalize", "mono", "eval_crop_mode",
)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Embedding cache CSV is empty: {path}")
    return rows


def row_audio_path(row: Mapping[str, str]) -> str:
    value = row.get("full_path") or row.get("path") or row.get("filepath") or ""
    if not value:
        raise ValueError("CSV row has no full_path/path/filepath")
    return str(value)


def _path_tree_fingerprint(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.exists():
        return {"path": public_path(path), "missing": True}
    if path.is_file():
        stat = path.stat()
        return {"path": public_path(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    digest = hashlib.sha256()
    count = 0
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = child.stat()
        digest.update(
            f"{child.relative_to(path).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
        )
        count += 1
    return {"path": public_path(path), "tree_stat_sha256": digest.hexdigest(), "files": count}


def encoder_cache_context(cfg: Mapping[str, Any], crop_mode: str) -> dict[str, Any]:
    branches = {
        name: value for name, value in cfg["audio_branches"].items() if value.get("enabled")
    }
    data_cfg = dict(cfg.get("data", {}))
    preprocess = {key: data_cfg.get(key) for key in PREPROCESS_KEYS}
    preprocess["eval_crop_mode"] = str(crop_mode)
    return {
        "preprocess": preprocess,
        "audio_branches": branches,
        "encoder_compatibility": dict(cfg.get("encoder_compatibility", {})),
        "encoder_weights": {
            name: _path_tree_fingerprint(value["weights"])
            for name, value in branches.items() if value.get("weights")
        },
    }


def _source_set_fingerprint(rows: list[dict[str, str]]) -> str:
    """CACHE_VERSION=1 source identity contract.

    This exact declared-hash plus file-stat representation is already embedded
    in existing R0 caches.  Any semantic change requires a cache-version
    bump and an explicit migration path.
    """
    digest = hashlib.sha256()
    for row in rows:
        path = Path(row_audio_path(row)).expanduser()
        declared = row.get("source_sha256") or row.get("audio_sha256")
        if path.is_file():
            stat = path.stat()
            identity = (
                f"declared:{declared or ''}:stat:{public_path(path)}:"
                f"{stat.st_size}:{stat.st_mtime_ns}"
            )
        else:
            identity = f"declared:{declared or ''}:missing:{public_path(path)}"
        digest.update(f"{identity}\n".encode())
    return digest.hexdigest()


def cache_spec(csv_path: str | Path, cfg: Mapping[str, Any], crop_mode: str) -> dict[str, Any]:
    rows = read_csv_rows(csv_path)
    return {
        "cache_version": CACHE_VERSION,
        "csv": public_path(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "rows": len(rows),
        "source_set_sha256": _source_set_fingerprint(rows),
        "crop_mode": str(crop_mode),
        **encoder_cache_context(cfg, crop_mode),
    }


def spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.no_grad()
def build_csv_embedding_cache(
    encoders: Mapping[str, torch.nn.Module],
    csv_path: str | Path,
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    crop_mode: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    resolved = deepcopy(dict(cfg))
    resolved["data"] = dict(resolved["data"])
    resolved["data"]["eval_crop_mode"] = str(crop_mode)
    resolved["data"]["auto_convert_mp3"] = False
    resolved["data"]["audio_cache_dir"] = None
    loader = make_audio_loader(
        str(csv_path), resolved, train=False, batch_size=batch_size,
        num_workers=num_workers, drop_last=False,
    )
    values: dict[str, list[torch.Tensor]] = {name: [] for name in encoders}
    labels: list[torch.Tensor] = []
    for batch in loader:
        labels.append(batch["label"].detach().cpu())
        for name, encoder in encoders.items():
            embedding = encoder(
                batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
                sample_rate=encoder.target_sr,
            )
            values[name].append(embedding.detach().cpu())
    if not labels:
        raise ValueError("Embedding cache loader produced zero samples")
    embeddings = {name: torch.cat(parts) for name, parts in values.items()}
    labels_tensor = torch.cat(labels).to(torch.long)
    rows = read_csv_rows(csv_path)
    if labels_tensor.numel() != len(rows):
        raise ValueError("CSV row count and cached embedding count differ")
    for name, tensor in embeddings.items():
        validate_embedding_tensor(tensor, f"embeddings[{name}]")
    spec = cache_spec(csv_path, resolved, crop_mode)
    return {
        "project": "CARLA", "kind": "csv_frozen_encoder_embeddings",
        "cache_version": CACHE_VERSION, "spec": spec,
        "signature_sha256": spec_sha256(spec), "embeddings": embeddings,
        "labels": labels_tensor, "rows": [publicize_row(row) for row in rows],
    }


def save_embedding_cache(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    torch_save_atomic(destination, dict(payload))
    stat = destination.stat()
    write_json_atomic(embedding_cache_metadata_path(destination), {
        "project": payload.get("project"),
        "kind": payload.get("kind"),
        "cache_version": payload.get("cache_version"),
        "signature_sha256": payload.get("signature_sha256"),
        "spec": payload.get("spec"),
        "artifact_size": stat.st_size,
        "artifact_mtime_ns": stat.st_mtime_ns,
    })


def embedding_cache_metadata_path(path: str | Path) -> Path:
    destination = Path(path)
    return destination.with_name(f"{destination.name}.meta.json")


def embedding_cache_artifact_sha256(path: str | Path) -> str:
    destination = Path(path)
    metadata_path = embedding_cache_metadata_path(destination)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Embedding cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stat = destination.stat()
    if (
        int(metadata.get("artifact_size", -1)) != stat.st_size
        or int(metadata.get("artifact_mtime_ns", -1)) != stat.st_mtime_ns
    ):
        raise ValueError(f"Embedding cache artifact changed after metadata write: {destination}")
    digest = metadata.get("artifact_sha256")
    if digest:
        return str(digest)
    digest = sha256_file(destination)
    metadata["artifact_sha256"] = digest
    write_json_atomic(metadata_path, metadata)
    return digest


def validate_stage1_randaugment_cache_metadata(
    path: str | Path, *, csv_path: str | Path, epoch: int, seed: int, split: str
) -> str:
    """Validate a shared R1 pair cache without loading its tensors."""

    destination = Path(path)
    metadata_path = embedding_cache_metadata_path(destination)
    if not destination.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Stage 1 cache or sidecar is missing: {destination}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spec = metadata.get("spec", {})
    expected = {
        "kind": "stage1_epoch_randaugment_embeddings",
        "split": str(split), "epoch": int(epoch), "seed": int(seed),
        "csv_sha256": sha256_file(csv_path),
    }
    actual = {key: spec.get(key) for key in expected}
    if metadata.get("project") != "CARLA" or metadata.get("kind") != expected["kind"]:
        raise ValueError(f"Not a CARLA Stage 1 pair cache: {destination}")
    if actual != expected:
        raise ValueError(
            f"Stage 1 cache provenance mismatch: expected={expected}, actual={actual}"
        )
    if int(metadata.get("artifact_size", -1)) != destination.stat().st_size:
        raise ValueError(f"Stage 1 cache artifact size changed: {destination}")
    if int(metadata.get("artifact_mtime_ns", -1)) != destination.stat().st_mtime_ns:
        raise ValueError(f"Stage 1 cache artifact mtime changed: {destination}")
    if not metadata.get("artifact_sha256"):
        raise ValueError(f"Stage 1 cache artifact SHA256 is missing: {destination}")
    signature = str(metadata.get("signature_sha256", ""))
    if not signature or signature != spec_sha256(spec):
        raise ValueError(f"Stage 1 cache signature mismatch: {destination}")
    return signature


def validate_csv_embedding_cache_metadata(
    path: str | Path, *, csv_path: str | Path, cfg: Mapping[str, Any], crop_mode: str
) -> dict[str, Any]:
    destination = Path(path)
    metadata_path = embedding_cache_metadata_path(destination)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stat = destination.stat()
        expected = spec_sha256(cache_spec(csv_path, cfg, crop_mode))
        if (
            metadata.get("project") != "CARLA"
            or metadata.get("kind") != "csv_frozen_encoder_embeddings"
            or int(metadata.get("cache_version", -1)) != CACHE_VERSION
            or metadata.get("signature_sha256") != expected
            or int(metadata.get("artifact_size", -1)) != stat.st_size
            or int(metadata.get("artifact_mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise ValueError(f"Embedding cache metadata mismatch: {destination}")
        return metadata
    payload = load_csv_embedding_cache(
        destination, csv_path=csv_path, cfg=cfg, crop_mode=crop_mode
    )
    stat = destination.stat()
    metadata = {
        "project": payload["project"], "kind": payload["kind"],
        "cache_version": payload["cache_version"],
        "signature_sha256": payload["signature_sha256"], "spec": payload["spec"],
        "artifact_size": stat.st_size, "artifact_mtime_ns": stat.st_mtime_ns,
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def load_csv_embedding_cache(
    path: str | Path,
    *,
    csv_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    crop_mode: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("project") != "CARLA" or payload.get("kind") != "csv_frozen_encoder_embeddings":
        raise ValueError(f"Not a CARLA CSV embedding cache: {path}")
    if int(payload.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported embedding cache version: {payload.get('cache_version')}")
    actual = spec_sha256(payload.get("spec", {}))
    if actual != payload.get("signature_sha256"):
        raise ValueError("Embedding cache metadata signature is corrupt")
    if csv_path is not None or cfg is not None or crop_mode is not None:
        if csv_path is None or cfg is None or crop_mode is None:
            raise ValueError("csv_path, cfg and crop_mode are required together for cache validation")
        expected = spec_sha256(cache_spec(csv_path, cfg, crop_mode))
        if expected != actual:
            raise ValueError(
                f"Embedding cache signature mismatch: expected={expected} actual={actual}"
            )
    embeddings = payload.get("embeddings", {})
    labels = payload.get("labels")
    if not embeddings or not isinstance(labels, torch.Tensor):
        raise ValueError("Embedding cache has no embeddings/labels")
    expected_branches = set(payload["spec"].get("audio_branches", {}))
    if set(embeddings) != expected_branches:
        raise ValueError(
            f"Embedding cache branch mismatch: expected={sorted(expected_branches)} "
            f"actual={sorted(embeddings)}"
        )
    sizes = {tensor.shape[0] for tensor in embeddings.values()} | {labels.shape[0]}
    if len(sizes) != 1:
        raise ValueError("Embedding cache row counts differ")
    expected_rows = int(payload["spec"].get("rows", -1))
    if labels.shape[0] != expected_rows or len(payload.get("rows", [])) != expected_rows:
        raise ValueError("Embedding cache row count does not match signed metadata")
    for name, tensor in embeddings.items():
        validate_embedding_tensor(tensor, f"embeddings[{name}]")
    return payload


class CsvEmbeddingCacheDataset(Dataset):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.embeddings = dict(payload["embeddings"])
        self.labels = payload["labels"].to(torch.long)
        self.rows = list(payload.get("rows", [{} for _ in range(len(self.labels))]))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "embeddings": {name: tensor[index] for name, tensor in self.embeddings.items()},
            "label": self.labels[index], "row_index": index,
        }


def make_embedding_cache_loader(
    path: str | Path,
    *,
    csv_path: str | Path,
    cfg: Mapping[str, Any],
    crop_mode: str,
    train: bool,
    batch_size: int,
    num_workers: int,
    drop_last: bool | None = None,
) -> DataLoader:
    payload = load_csv_embedding_cache(
        path, csv_path=csv_path, cfg=cfg, crop_mode=crop_mode,
    )
    return DataLoader(
        CsvEmbeddingCacheDataset(payload), batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=configure_data_worker_threads if num_workers > 0 else None,
        drop_last=train if drop_last is None else bool(drop_last),
    )
