"""Small, dependency-free experiment provenance and logging helpers."""
from __future__ import annotations

import json
import os
import platform
import hashlib
import random
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def atomic_output_path(path: str | Path):
    """Yield a unique same-directory temporary path and atomically publish it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def unique_temporary_path(path: str | Path) -> Path:
    """Reserve and return a unique same-directory temporary path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    with atomic_output_path(destination) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def torch_save_atomic(path: str | Path, value: Any) -> None:
    with atomic_output_path(path) as temporary:
        torch.save(value, temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    """Return a portable, non-identifying path label for runtime metadata."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        return value.as_posix()
    resolved = value.resolve()
    roots = []
    for variable, label in (
        ("CARLA_PROJECT_ROOT", "project"),
        ("CARLA_DATA_ROOT", "data"),
        ("SOFIA_ROOT", "sofia"),
    ):
        configured = os.environ.get(variable)
        if configured:
            roots.append((Path(configured).expanduser().resolve(), label))
    roots.append((Path(__file__).resolve().parents[2], "project"))
    for root_path, label in roots:
        try:
            relative = resolved.relative_to(root_path)
        except ValueError:
            continue
        return f"{label}/{relative.as_posix()}" if str(relative) != "." else label
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return f"external/{digest}/{resolved.name}"


def publicize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Redact local filesystem locations in CSV-derived runtime records."""
    output = dict(row)
    for key in ("full_path", "path", "filepath", "clean_path", "corrupt_path"):
        value = output.get(key)
        if isinstance(value, str) and value:
            output[key] = public_path(value)
    return output


def model_fingerprint(base_checkpoint: str | Path, lora_checkpoint: str | Path | None = None) -> str:
    """Hash the ordered model artifact identities used for one inference run."""
    digest = hashlib.sha256()
    digest.update(f"base:{sha256_file(base_checkpoint)}\n".encode())
    if lora_checkpoint is not None:
        digest.update(f"lora:{sha256_file(lora_checkpoint)}\n".encode())
    return digest.hexdigest()


def bank_fingerprint(base_checkpoint: str | Path, lora_checkpoints: list[str | Path]) -> str:
    """Hash one ordered cumulative bank; order is part of the deployed model."""
    return bank_fingerprint_from_hashes(
        sha256_file(base_checkpoint),
        [sha256_file(path) for path in lora_checkpoints],
    )


def bank_fingerprint_from_hashes(base_sha256: str, lora_sha256: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"base:{base_sha256}\n".encode())
    for index, value in enumerate(lora_sha256):
        digest.update(f"lora[{index}]:{value}\n".encode())
    return digest.hexdigest()


def source_tree_fingerprint(root: str | Path | None = None) -> dict[str, Any]:
    project = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    files: set[Path] = set(project.glob("*.py"))
    for directory, suffixes in (("freya_pro", {".py"}), ("scripts", {".py", ".sh"}),
                                ("config", {".yaml", ".yml"})):
        base = project / directory
        if base.is_dir():
            files.update(path for path in base.rglob("*") if path.is_file() and path.suffix in suffixes)
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(project).as_posix()
        digest.update(f"{relative}:{sha256_file(path)}\n".encode())
    return {"root": public_path(project), "sha256": digest.hexdigest(), "file_count": len(files)}


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def environment_manifest() -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "argv": [public_path(value) if Path(value).is_absolute() else value for value in sys.argv],
        "python": sys.version,
        "python_executable": public_path(sys.executable),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": public_path(os.environ["CONDA_PREFIX"]) if os.environ.get("CONDA_PREFIX") else None,
        "source_tree": source_tree_fingerprint(),
    }


def start_telemetry(device: torch.device) -> dict[str, Any]:
    cuda_index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None
    memory_error = None
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(cuda_index)
        except RuntimeError as exc:
            memory_error = repr(exc)
    return {"started_utc": utc_now(), "monotonic_start": time.perf_counter(),
            "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_index": cuda_index,
            "cuda_memory_telemetry_error": memory_error,
            "cuda_device_name": torch.cuda.get_device_name(cuda_index) if device.type == "cuda" else None}


def telemetry_snapshot(state: dict[str, Any], device: torch.device) -> dict[str, Any]:
    value = {"wall_seconds": float(time.perf_counter() - state["monotonic_start"]),
             "device": str(device), "cuda_visible_devices": state.get("cuda_visible_devices"),
             "cuda_device_name": state.get("cuda_device_name")}
    if device.type == "cuda":
        cuda_index = state["cuda_index"]
        value["cuda_index"] = cuda_index
        try:
            value.update({"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index)),
                          "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_index))})
        except RuntimeError as exc:
            value["cuda_memory_telemetry_error"] = repr(exc)
    return value


def prune_epoch_checkpoints(directory: str | Path, pattern: str, keep: int = 1) -> list[str]:
    """Remove only older numbered resume checkpoints; best/final files never match the pattern."""
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime_ns)
    removed = []
    for path in files[:-max(int(keep), 1)]:
        path.unlink()
        removed.append(str(path))
    return removed


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        saved_cuda = list(state["cuda"])
        visible_cuda = torch.cuda.device_count()
        if len(saved_cuda) == visible_cuda:
            torch.cuda.set_rng_state_all(saved_cuda)
        else:
            warnings.warn(
                "CUDA RNG state count differs from the checkpoint; restoring only "
                f"the first {min(len(saved_cuda), visible_cuda)} visible GPU state(s) "
                f"(checkpoint={len(saved_cuda)}, visible={visible_cuda}).",
                RuntimeWarning,
            )
            for device_index, rng_state in enumerate(saved_cuda[:visible_cuda]):
                torch.cuda.set_rng_state(rng_state, device=device_index)
