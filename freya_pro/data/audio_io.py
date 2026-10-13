"""Lazy audio decoding and MP3-to-WAV cache helpers.

The cache is populated on first access rather than by materializing an entire
dataset.  WAV files are written outside the repository by default, so a large
MP3 collection does not create a second checked-in dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from ..utils.run_io import atomic_output_path, public_path

try:
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover - tensor-only tests
    torchaudio = None


COMPRESSED_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".opus"}
CACHE_VERSION = 2
DEFAULT_LOCK_TIMEOUT_SECONDS = 900.0
STALE_LOCK_SECONDS = 3600.0


def _require_torchaudio():
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to decode audio files")
    return torchaudio


def _resolve_ffmpeg() -> str | None:
    """Resolve the launcher-provided ffmpeg executable for decode fallback."""

    candidates = [
        os.environ.get("CARLA_FFMPEG_BIN", ""),
        shutil.which("ffmpeg"),
        str(Path(sys.executable).resolve().parent / "ffmpeg"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _project_root() -> Path:
    """Return the persistent project/personal root for generated artifacts."""

    configured = os.environ.get("CARLA_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser()
    # audio_io.py lives at <project>/freya_pro/data/audio_io.py.
    return Path(__file__).resolve().parents[2]


def _default_runtime_tmpdir() -> Path:
    """Return a project-local temporary directory, never container /tmp."""

    configured = os.environ.get("CARLA_TMPDIR")
    root = Path(configured).expanduser() if configured else _project_root() / ".tmp" / "freya_pro"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_with_ffmpeg(path: Path, audio) -> tuple[torch.Tensor, int]:
    """Decode one source through ffmpeg when the torchaudio backend fails.

    Some Jamendo MP3 files are accepted by ffmpeg but rejected by the
    runtime's libsndfile-backed torchaudio reader. The fallback is bounded,
    one-threaded per worker, and only runs after the normal reader fails.
    """

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable for audio decode fallback")
    temp_root = str(_default_runtime_tmpdir())
    timeout = float(os.environ.get("CARLA_AUDIO_DECODE_TIMEOUT", "120"))
    with tempfile.TemporaryDirectory(prefix="freya_decode_", dir=temp_root) as directory:
        decoded = Path(directory) / "decoded.wav"
        command = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-threads", "1", "-i", str(path), "-vn",
            "-c:a", "pcm_f32le", str(decoded),
        ]
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"ffmpeg decode timed out after {timeout:g}s"
            ) from error
        if completed.returncode != 0 or not decoded.is_file():
            detail = (completed.stderr or "").strip().replace("\n", " ")
            raise RuntimeError(
                f"ffmpeg decode failed with exit={completed.returncode}: {detail[:500]}"
            )
        return audio.load(str(decoded))


def default_audio_cache_dir() -> Path:
    configured = os.getenv("CARLA_WAV_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_root = os.getenv("CARLA_CACHE_ROOT")
    if cache_root:
        return Path(cache_root).expanduser() / "wav"
    # Use an explicit cache root or a project-local cache directory.
    return _project_root() / ".cache" / "freya_pro" / "wav"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": public_path(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_key(path: Path) -> str:
    payload = json.dumps(
        {"cache_version": CACHE_VERSION, "source": _source_fingerprint(path)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_wav_path(path: Path, cache_dir: Path) -> Path:
    digest = _cache_key(path)
    return cache_dir / digest[:2] / f"{digest}.wav"


def _metadata_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(".json")


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _cache_is_valid(source: Path, destination: Path, audio) -> bool:
    """Validate both provenance and a readable WAV header before reuse."""

    if not destination.is_file() or destination.stat().st_size <= 44:
        return False
    metadata = _read_metadata(_metadata_path(destination))
    if metadata is None:
        return False
    if metadata.get("cache_version") != CACHE_VERSION:
        return False
    if metadata.get("source") != _source_fingerprint(source):
        return False
    try:
        info = audio.info(str(destination))
        if int(getattr(info, "num_frames", 0)) <= 0:
            return False
        if int(getattr(info, "sample_rate", 0)) <= 0:
            return False
        if int(getattr(info, "num_channels", 0)) <= 0:
            return False
        return (
            int(metadata.get("num_frames", -1)) == int(info.num_frames)
            and int(metadata.get("sample_rate", -1)) == int(info.sample_rate)
            and int(metadata.get("num_channels", -1)) == int(info.num_channels)
        )
    except Exception:
        return False


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _acquire_lock(
    source: Path,
    destination: Path,
    lock_path: Path,
    audio,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> bool:
    """Acquire a per-source lock, or wait for another worker's valid cache."""

    deadline = time.monotonic() + float(timeout_seconds)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if _cache_is_valid(source, destination, audio):
            return False
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
            return True
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > STALE_LOCK_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for audio cache lock: {lock_path}")
            time.sleep(0.1)


def resolve_audio_path(
    path: str | Path,
    *,
    auto_convert_mp3: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[Path, bool]:
    """Resolve compressed input to a lazily materialized WAV path."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if not auto_convert_mp3 or source.suffix.lower() not in COMPRESSED_EXTENSIONS:
        return source, False

    audio = _require_torchaudio()
    root = Path(cache_dir).expanduser() if cache_dir else default_audio_cache_dir()
    destination = _cached_wav_path(source, root)
    if _cache_is_valid(source, destination, audio):
        return destination, True

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(".lock")
    acquired = _acquire_lock(source, destination, lock_path, audio)
    if not acquired:
        return destination, True
    try:
        try:
            waveform, sample_rate = audio.load(str(source))
        except Exception as primary_error:
            try:
                waveform, sample_rate = _load_with_ffmpeg(source, audio)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Unable to decode audio path={source}: "
                    f"primary={type(primary_error).__name__}: {primary_error}; "
                    f"ffmpeg={type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error
        # A process-specific temporary file prevents a partially written WAV
        # from being observed by another DataLoader worker.
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=f".{destination.stem}.",
            suffix=".wav",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            audio.save(
                str(temporary),
                waveform.detach().cpu().to(torch.float32),
                int(sample_rate),
                format="wav",
                encoding="PCM_F",
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        info = audio.info(str(destination))
        _write_metadata(
            _metadata_path(destination),
            {
                "cache_version": CACHE_VERSION,
                "source": _source_fingerprint(source),
                "num_frames": int(info.num_frames),
                "sample_rate": int(info.sample_rate),
                "num_channels": int(info.num_channels),
                "encoding": "PCM_F",
            },
        )
    finally:
        lock_path.unlink(missing_ok=True)
    return destination, True


def load_audio_cached(
    path: str | Path,
    *,
    target_sr: int | None = None,
    mono: bool = True,
    normalize: bool = True,
    strict: bool = True,
    auto_convert_mp3: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio, lazily converting compressed sources to cached WAV."""

    audio = _require_torchaudio()
    try:
        resolved, _ = resolve_audio_path(
            path,
            auto_convert_mp3=auto_convert_mp3,
            cache_dir=cache_dir,
        )
        waveform, sample_rate = audio.load(str(resolved))
    except Exception as error:
        if strict:
            source = Path(path).expanduser()
            raise RuntimeError(
                f"Audio decode failed for path={public_path(source)}: "
                f"{type(error).__name__}: {error}"
            ) from error
        sample_rate = int(target_sr or 44100)
        waveform = torch.zeros(1, 1, dtype=torch.float32)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and int(sample_rate) != int(target_sr):
        waveform = audio.functional.resample(
            waveform, int(sample_rate), int(target_sr)
        )
        sample_rate = int(target_sr)
    if normalize:
        peak = waveform.abs().max().clamp(min=1e-9)
        waveform = waveform / peak
    return waveform, int(sample_rate)
