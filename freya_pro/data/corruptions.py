"""Full-song music corruptions and Audio RandAugment primitives.

The benchmark operates on float32 ``[channels, samples]`` waveforms.  The
functions in this module deliberately avoid detector-specific state: a
condition can be generated once and reused by different CARLA models.
Operations that require an external codec or resource are explicit failures
instead of silently substituting a different transformation.
"""
from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
try:
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover - optional for pure tensor tests
    torchaudio = None


FAMILIES = ("A", "B", "C", "D", "E")


def _require_torchaudio() -> Any:
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is required for filter, pitch/time, resampling and codec operations"
        )
    return torchaudio


@dataclass(frozen=True)
class OperationSpec:
    """A registry entry used by fixed conditions and RandAugment."""

    operation: str
    family: str
    parameter: str
    minimum: float
    maximum: float
    length_preserving: bool = True
    status: str = "core"
    resource_key: str | None = None


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Return a finite float32 stereo waveform without normalizing it."""

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] == 0 or waveform.shape[-1] == 0:
        raise ValueError("Empty waveform")
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains NaN or Inf")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.to(dtype=torch.float32)


def _generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return generator


def rms_db(waveform: torch.Tensor, eps: float = 1e-12) -> float:
    """Measure aggregate RMS level in dBFS."""

    value = waveform.float().square().mean().clamp_min(eps)
    return float(10.0 * torch.log10(value).item())


def snr_db(clean: torch.Tensor, corrupted: torch.Tensor, eps: float = 1e-12) -> float:
    """Measure aggregate SNR for equal-length signals."""

    length = min(clean.shape[-1], corrupted.shape[-1])
    signal = clean[..., :length].float()
    noise = corrupted[..., :length].float() - signal
    ratio = signal.square().mean() / noise.square().mean().clamp_min(eps)
    return float(10.0 * torch.log10(ratio).item())


def _db_to_amplitude(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def apply_gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    return ensure_stereo(waveform) * _db_to_amplitude(gain_db)


def peak_normalize(waveform: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    peak = waveform.abs().amax()
    if peak <= 1e-12:
        raise ValueError("Cannot peak-normalize an all-zero waveform")
    return waveform * (_db_to_amplitude(target_dbfs) / peak)


def loudness_normalize(waveform: torch.Tensor, target_lufs: float) -> torch.Tensor:
    """Apply a linear loudness proxy gain.

    A full BS.1770 meter is optional in the dataset builder.  The DSP
    operation itself intentionally remains linear, so it cannot accidentally
    introduce a limiter or compressor.  ``target_lufs`` is interpreted as the
    RMS proxy level when no external loudness meter is installed.
    """

    waveform = ensure_stereo(waveform)
    return apply_gain(waveform, float(target_lufs) - rms_db(waveform))


def _linked_envelope(waveform: torch.Tensor, window: int) -> torch.Tensor:
    window = max(int(window), 1)
    if window == 1:
        return waveform.abs().amax(dim=0)
    values = waveform.abs().mean(dim=0, keepdim=True).unsqueeze(0)
    padded = F.pad(values, (window // 2, window - 1 - window // 2), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1).sqrt().squeeze(0).squeeze(0)


def compress(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -18.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> torch.Tensor:
    """Apply a deterministic stereo-linked downward compressor."""

    del release_ms  # The envelope is intentionally stateless and full-song deterministic.
    waveform = ensure_stereo(waveform)
    if ratio < 1.0:
        raise ValueError("Compression ratio must be >= 1")
    window = max(round(float(attack_ms) * sample_rate / 1000.0), 1)
    envelope = _linked_envelope(waveform, window).clamp_min(1e-8)
    threshold = _db_to_amplitude(threshold_db)
    gain = torch.ones_like(envelope)
    above = envelope > threshold
    gain[above] = (threshold / envelope[above]).pow(1.0 - 1.0 / float(ratio))
    return waveform * gain.unsqueeze(0)


def limit(
    waveform: torch.Tensor,
    drive_db: float = 6.0,
    ceiling_dbtp: float = -1.0,
) -> torch.Tensor:
    """Apply a stereo-linked peak limiter with a fixed ceiling."""

    driven = apply_gain(waveform, drive_db)
    ceiling = _db_to_amplitude(ceiling_dbtp)
    peak = driven.abs().amax(dim=0).clamp_min(1e-8)
    gain = torch.minimum(torch.ones_like(peak), ceiling / peak)
    return driven * gain.unsqueeze(0)


def clip(waveform: torch.Tensor, drive_db: float = 6.0, soft: bool = False) -> torch.Tensor:
    driven = apply_gain(waveform, drive_db)
    if soft:
        scale = math.tanh(_db_to_amplitude(drive_db))
        return torch.tanh(driven * 3.0) / max(scale, 1e-6)
    return driven.clamp(-1.0, 1.0)


def fade(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    direction: str = "in",
    seconds: float = 1.0,
    shape: str = "half_sine",
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    count = min(max(round(float(seconds) * sample_rate), 1), waveform.shape[-1])
    phase = torch.linspace(0.0, 1.0, count, device=waveform.device)
    shape = shape.lower()
    if shape == "linear":
        envelope = phase
    elif shape == "logarithmic":
        envelope = torch.log1p(9.0 * phase) / math.log(10.0)
    elif shape == "exponential":
        envelope = (torch.exp(phase * math.log(10.0)) - 1.0) / 9.0
    elif shape == "quarter_sine":
        envelope = torch.sin(phase * math.pi / 2.0)
    elif shape == "half_sine":
        envelope = torch.sin(phase * math.pi / 2.0)
    else:
        raise ValueError(f"Unknown fade shape: {shape}")

    output = waveform.clone()
    if direction in {"in", "both"}:
        output[..., :count] *= envelope
    if direction in {"out", "both"}:
        output[..., -count:] *= envelope.flip(0)
    if direction not in {"in", "out", "both"}:
        raise ValueError(f"Unknown fade direction: {direction}")
    return output


def trim(waveform: torch.Tensor, sample_rate: int, *, direction: str, seconds: float) -> torch.Tensor:
    del sample_rate
    waveform = ensure_stereo(waveform)
    count = round(float(seconds) * 1.0)
    if count < 0:
        raise ValueError("Trim length must be non-negative")
    # ``seconds`` is converted by the caller so this helper accepts samples
    # through the private dispatch below.
    if direction == "head":
        return waveform[..., count:]
    if direction == "tail":
        return waveform[..., :-count] if count else waveform
    raise ValueError(f"Unknown trim direction: {direction}")


def trim_seconds(waveform: torch.Tensor, sample_rate: int, direction: str, seconds: float) -> torch.Tensor:
    count = round(float(seconds) * sample_rate)
    waveform = ensure_stereo(waveform)
    if count <= 0 or count >= waveform.shape[-1]:
        raise ValueError("Trim length must be positive and shorter than the song")
    if direction == "head":
        return waveform[..., count:]
    if direction == "tail":
        return waveform[..., :-count]
    raise ValueError(f"Unknown trim direction: {direction}")


def zero_pad(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    direction: str = "tail",
    seconds: float = 0.5,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    count = max(round(float(seconds) * sample_rate), 0)
    zeros = torch.zeros((2, count), dtype=waveform.dtype, device=waveform.device)
    if direction == "head":
        return torch.cat((zeros, waveform), dim=-1)
    if direction == "tail":
        return torch.cat((waveform, zeros), dim=-1)
    if direction == "both":
        return torch.cat((zeros, waveform, zeros), dim=-1)
    raise ValueError(f"Unknown padding direction: {direction}")


def dropout(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    missing_ratio: float = 0.005,
    frame_ms: float = 30.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError("missing_ratio must be in [0, 1]")
    frame = max(round(float(frame_ms) * sample_rate / 1000.0), 1)
    target = round(waveform.shape[-1] * float(missing_ratio))
    count = max(0, math.ceil(target / frame))
    rng = random.Random(seed)
    mask = torch.ones(waveform.shape[-1], dtype=waveform.dtype, device=waveform.device)
    occupied: list[tuple[int, int]] = []
    attempts = 0
    while len(occupied) < count and attempts < max(count * 20, 20):
        attempts += 1
        start = rng.randint(0, max(waveform.shape[-1] - frame, 0))
        end = min(start + frame, waveform.shape[-1])
        if any(start < old_end and end > old_start for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        mask[start:end] = 0.0
    return waveform * mask.unsqueeze(0)


def add_white_noise(
    waveform: torch.Tensor,
    snr_db_value: float,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    generator = _generator(seed, waveform.device)
    noise = torch.randn(
        waveform.shape,
        dtype=waveform.dtype,
        device=waveform.device,
        generator=generator,
    )
    signal_power = waveform.square().mean()
    noise_power = signal_power / (10.0 ** (float(snr_db_value) / 10.0))
    noise = noise * torch.sqrt(noise_power / noise.square().mean().clamp_min(1e-12))
    return waveform + noise


def add_pink_noise(
    waveform: torch.Tensor,
    snr_db_value: float,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    generator = _generator(seed, waveform.device)
    white = torch.randn(
        waveform.shape,
        dtype=waveform.dtype,
        device=waveform.device,
        generator=generator,
    )
    spectrum = torch.fft.rfft(white, dim=-1)
    frequencies = torch.arange(
        spectrum.shape[-1], device=waveform.device, dtype=waveform.dtype
    ).clamp_min(1.0)
    spectrum = spectrum / frequencies.sqrt()
    noise = torch.fft.irfft(spectrum, n=waveform.shape[-1], dim=-1).real
    signal_power = waveform.square().mean()
    noise_power = signal_power / (10.0 ** (float(snr_db_value) / 10.0))
    noise = noise * torch.sqrt(noise_power / noise.square().mean().clamp_min(1e-12))
    return waveform + noise


def add_environmental_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    *,
    snr_db_value: float = 20.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    noise = ensure_stereo(noise)
    if noise.shape[-1] != waveform.shape[-1]:
        if noise.shape[-1] < waveform.shape[-1]:
            repeats = math.ceil(waveform.shape[-1] / noise.shape[-1])
            noise = noise.repeat(1, repeats)[..., : waveform.shape[-1]]
        else:
            generator = random.Random(seed)
            start = generator.randint(0, noise.shape[-1] - waveform.shape[-1])
            noise = noise[..., start : start + waveform.shape[-1]]
    del sample_rate
    signal_power = waveform.square().mean()
    target_power = signal_power / (10.0 ** (float(snr_db_value) / 10.0))
    noise = noise * torch.sqrt(target_power / noise.square().mean().clamp_min(1e-12))
    return waveform + noise


def add_echo(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    delay_ms: float = 300.0,
    decay: float = 0.35,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    delay = max(round(float(delay_ms) * sample_rate / 1000.0), 1)
    output = F.pad(waveform, (0, delay))
    output[..., delay:] += waveform * float(decay)
    return output


def add_rir(
    waveform: torch.Tensor,
    rir: torch.Tensor,
    *,
    dry_wet: float = 0.35,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform)
    rir = ensure_stereo(rir)
    if rir.shape[0] > 1:
        rir = rir.mean(dim=0, keepdim=True)
    rir = rir.squeeze(0)
    rir = rir / rir.square().sum().sqrt().clamp_min(1e-8)
    # A 60 s model window convolved with a roughly 3 s Rochester impulse
    # response is prohibitively expensive with direct conv1d (O(N*M)).
    # Linear FFT convolution is mathematically equivalent to the former full
    # convolution while reducing the cost to O((N+M) log(N+M)).  Padding to a
    # power of two also prevents circular wrap-around and gives predictable
    # CPU FFT performance in spawned data-loader workers.
    full_length = int(waveform.shape[-1] + rir.numel() - 1)
    fft_length = 1 << (full_length - 1).bit_length()
    signal_spectrum = torch.fft.rfft(waveform, n=fft_length, dim=-1)
    rir_spectrum = torch.fft.rfft(rir, n=fft_length, dim=-1)
    convolved = torch.fft.irfft(
        signal_spectrum * rir_spectrum.unsqueeze(0),
        n=fft_length,
        dim=-1,
    )[..., :full_length]
    dry = F.pad(waveform, (0, convolved.shape[-1] - waveform.shape[-1]))
    wet = convolved
    mix = float(dry_wet)
    return dry * (1.0 - mix) + wet * mix


def equalizer(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    kind: str = "bell",
    frequency_hz: float = 3000.0,
    gain_db: float = 3.0,
    q: float = 1.0,
) -> torch.Tensor:
    audio = _require_torchaudio()
    waveform = ensure_stereo(waveform)
    kind = kind.lower()
    if kind == "low_shelf":
        return audio.functional.bass_biquad(
            waveform, sample_rate, gain_db, frequency_hz, q
        )
    if kind == "high_shelf":
        return audio.functional.treble_biquad(
            waveform, sample_rate, gain_db, frequency_hz, q
        )
    if kind == "bell":
        return audio.functional.equalizer_biquad(
            waveform, sample_rate, frequency_hz, gain_db, q
        )
    raise ValueError(f"Unknown EQ kind: {kind}")


def lowpass(waveform: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    audio = _require_torchaudio()
    return audio.functional.lowpass_biquad(
        ensure_stereo(waveform), sample_rate, float(cutoff_hz)
    )


def highpass(waveform: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    audio = _require_torchaudio()
    return audio.functional.highpass_biquad(
        ensure_stereo(waveform), sample_rate, float(cutoff_hz)
    )


def bandpass(
    waveform: torch.Tensor,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    audio = _require_torchaudio()
    return audio.functional.bandpass_biquad(
        ensure_stereo(waveform),
        sample_rate,
        central_freq=(float(low_hz) + float(high_hz)) / 2.0,
        Q=(float(low_hz) + float(high_hz)) / max(float(high_hz) - float(low_hz), 1.0),
    )


def bandstop(
    waveform: torch.Tensor,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    audio = _require_torchaudio()
    center = (float(low_hz) + float(high_hz)) / 2.0
    q = center / max(float(high_hz) - float(low_hz), 1.0)
    return audio.functional.bandreject_biquad(
        ensure_stereo(waveform), sample_rate, center, q
    )


def temporal_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    shift_fraction: float = 0.0,
    *,
    rollover: bool = True,
) -> torch.Tensor:
    """Shift samples in time; this is audiomentations ``Shift``, not pitch shift."""

    del sample_rate
    waveform = ensure_stereo(waveform)
    if not -1.0 <= float(shift_fraction) <= 1.0:
        raise ValueError("shift_fraction must be in [-1, 1]")
    count = int(round(float(shift_fraction) * waveform.shape[-1]))
    output = torch.roll(waveform, shifts=count, dims=-1)
    if not rollover and count > 0:
        output[..., :count] = 0.0
    elif not rollover and count < 0:
        output[..., count:] = 0.0
    return output


def pitch_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    semitones: float,
    *,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> torch.Tensor:
    audio = _require_torchaudio()
    waveform = ensure_stereo(waveform)
    n_fft = int(n_fft or os.environ.get("FREYA_PITCH_SHIFT_N_FFT", "1024"))
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    win_length = win_length or n_fft
    hop_length = hop_length or n_fft // 4
    # The R1 RandAugment window is already model-native in duration. Use the
    # FREYA CPU setting (1024/256) so pitchshift cannot dominate a loader
    # worker; the operation remains torchaudio's length-preserving pitch shift.
    with torch.no_grad():
        return audio.functional.pitch_shift(
            waveform,
            sample_rate,
            float(semitones),
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )


def time_stretch(
    waveform: torch.Tensor,
    sample_rate: int,
    speed_factor: float,
    *,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
) -> torch.Tensor:
    """Pitch-preserving phase-vocoder time stretch."""

    audio = _require_torchaudio()
    del sample_rate
    waveform = ensure_stereo(waveform)
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    hop_length = hop_length or n_fft // 4
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0.0,
        math.pi * hop_length,
        spec.shape[-2],
        device=waveform.device,
        dtype=waveform.dtype,
    )[..., None]
    stretched = audio.functional.phase_vocoder(
        spec, float(speed_factor), phase_advance
    )
    output_length = max(round(waveform.shape[-1] / float(speed_factor)), 1)
    return torch.istft(
        stretched,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=output_length,
    )


def resample_roundtrip(
    waveform: torch.Tensor,
    native_sample_rate: int,
    intermediate_sample_rate: int,
) -> torch.Tensor:
    audio = _require_torchaudio()
    waveform = ensure_stereo(waveform)
    down = audio.functional.resample(
        waveform, native_sample_rate, int(intermediate_sample_rate)
    )
    return audio.functional.resample(
        down, int(intermediate_sample_rate), native_sample_rate
    )


def quantize(
    waveform: torch.Tensor,
    bits: int,
    *,
    dither: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    waveform = ensure_stereo(waveform).clamp(-1.0, 1.0)
    levels = float(2 ** int(bits) - 1)
    if bits <= 0:
        raise ValueError("bits must be positive")
    value = waveform
    if dither:
        generator = _generator(seed, waveform.device)
        value = value + (
            torch.rand(
                value.shape,
                device=value.device,
                dtype=value.dtype,
                generator=generator,
            )
            - 0.5
        ) / levels
    return torch.round(value * levels) / levels


def codec_roundtrip(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    codec: str = "mp3",
    bitrate_kbps: int = 128,
    ffmpeg_bin: str = "ffmpeg",
) -> torch.Tensor:
    """Encode/decode one full song through an explicit FFmpeg codec.

    The external executable and codec name are part of the condition metadata.
    Missing FFmpeg is an error, never a fallback to quantization or resampling.
    """

    audio = _require_torchaudio()
    waveform = ensure_stereo(waveform).detach().cpu()
    ffmpeg = shutil.which(str(ffmpeg_bin))
    if ffmpeg is None and str(ffmpeg_bin) == "ffmpeg":
        # Direct Python invocations and spawned workers may not inherit an
        # activated Conda PATH. Prefer the launcher-provided absolute path,
        # then the bin directory next to the running Python interpreter.
        candidates = [
            os.environ.get("FREYA_FFMPEG_BIN", ""),
            str(Path(sys.executable).resolve().parent / "ffmpeg"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
                ffmpeg = candidate
                break
    if ffmpeg is None:
        raise RuntimeError(f"FFmpeg executable not found: {ffmpeg_bin}")
    codec = codec.lower()
    codec_map = {
        "mp3": ("libmp3lame", ".mp3"),
        "aac": ("aac", ".m4a"),
        "opus": ("libopus", ".ogg"),
    }
    if codec not in codec_map:
        raise ValueError(f"Unsupported codec: {codec}")
    encoder, suffix = codec_map[codec]
    with tempfile.TemporaryDirectory(prefix="freya_codec_") as directory:
        directory = Path(directory)
        source = directory / "input.wav"
        target = directory / f"output{suffix}"
        decoded = directory / "decoded.wav"
        audio.save(str(source), waveform, int(sample_rate), encoding="PCM_F")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-c:a",
            encoder,
            "-b:a",
            f"{int(bitrate_kbps)}k",
            str(target),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        decode_command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(target), "-vn", "-ar", str(int(sample_rate)),
            "-ac", "2", "-c:a", "pcm_f32le", str(decoded),
        ]
        subprocess.run(decode_command, check=True, capture_output=True, timeout=120)
        output, output_sr = audio.load(str(decoded))
    output = ensure_stereo(output)
    if int(output_sr) != int(sample_rate):
        output = audio.functional.resample(output, int(output_sr), int(sample_rate))
    return output


@lru_cache(maxsize=2)
def _load_encodec_model(weights: str, device: str):
    try:
        from encodec import EncodecModel
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("Install the official 'encodec' package") from exc
    model = EncodecModel.encodec_model_48khz(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(torch.device(device))


def encodec_roundtrip(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    weights: str,
    bandwidth_kbps: float = 6.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Official Meta EnCodec 48 kHz stereo encode/decode using local weights."""
    from encodec.utils import convert_audio

    source = ensure_stereo(waveform).detach()
    target_device = torch.device(device)
    model = _load_encodec_model(str(Path(weights).resolve()), str(target_device))
    model.set_target_bandwidth(float(bandwidth_kbps))
    value = convert_audio(source, int(sample_rate), model.sample_rate, model.channels)
    with torch.inference_mode():
        frames = model.encode(value.unsqueeze(0).to(target_device))
        decoded = model.decode(frames)[0].cpu()
    decoded = convert_audio(decoded, model.sample_rate, int(sample_rate), 2)
    length = source.shape[-1]
    return F.pad(decoded[..., :length], (0, max(length - decoded.shape[-1], 0)))


@lru_cache(maxsize=2)
def _load_dac_model(weights: str, device: str):
    try:
        import dac
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("Install the official 'descript-audio-codec' package") from exc
    return dac.DAC.load(weights).eval().to(torch.device(device))


def dac_roundtrip(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    weights: str,
    n_quantizers: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Official Descript DAC round-trip; stereo channels are a mono batch."""
    source = ensure_stereo(waveform).detach()
    target_device = torch.device(device)
    model = _load_dac_model(str(Path(weights).resolve()), str(target_device))
    with torch.inference_mode():
        result = model(
            source.unsqueeze(1).to(target_device), sample_rate=int(sample_rate),
            n_quantizers=n_quantizers,
        )
    decoded = result["audio"].squeeze(1).cpu()
    length = source.shape[-1]
    return F.pad(decoded[..., :length], (0, max(length - decoded.shape[-1], 0)))


def packet_loss(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    loss_rate: float = 0.05,
    mode: str = "hold",
    burst: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simulate packet-domain loss after packetization.

    This is intentionally named ``packet_loss`` rather than ``opus``: it is a
    deterministic PLC surrogate and must not be reported as a real codec
    encode/decode unless an Opus backend is used by the caller.
    """

    waveform = ensure_stereo(waveform)
    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError("loss_rate must be in [0, 1]")
    frame = max(round(float(frame_ms) * sample_rate / 1000.0), 1)
    output = waveform.clone()
    rng = random.Random(seed)
    previous = output[..., :frame].clone()
    state = False
    for start in range(0, waveform.shape[-1], frame):
        end = min(start + frame, waveform.shape[-1])
        if burst:
            if state:
                lost = rng.random() < min(1.0, loss_rate * 2.0)
            else:
                lost = rng.random() < loss_rate
            state = lost
        else:
            lost = rng.random() < loss_rate
        if not lost:
            previous = waveform[..., start:end].clone()
            continue
        if mode == "zero":
            output[..., start:end] = 0.0
        elif mode == "hold":
            output[..., start:end] = previous[..., : end - start]
        elif mode == "linear":
            right = waveform[..., min(end, waveform.shape[-1] - 1)]
            left = previous[..., -1]
            steps = end - start
            alpha = torch.linspace(
                0.0, 1.0, steps, device=waveform.device, dtype=waveform.dtype
            )
            output[..., start:end] = left.unsqueeze(-1) * (1.0 - alpha) + right.unsqueeze(-1) * alpha
        else:
            raise ValueError(f"Unknown PLC mode: {mode}")
    return output


def apply_operation(
    waveform: torch.Tensor,
    sample_rate: int,
    operation: str,
    parameters: Optional[Mapping[str, Any]] = None,
    *,
    resources: Optional[Mapping[str, torch.Tensor]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch one registered operation."""

    parameters = dict(parameters or {})
    resources = resources or {}
    operation = operation.lower().replace("-", "_")
    if operation == "gain":
        return apply_gain(waveform, parameters.get("gain_db", -6.0))
    if operation == "peak_normalization":
        return peak_normalize(waveform, parameters.get("target_dbfs", -3.0))
    if operation == "loudness_normalization":
        return loudness_normalize(waveform, parameters.get("target_lufs", -16.0))
    if operation == "compression":
        return compress(waveform, sample_rate, **parameters)
    if operation == "limiter":
        return limit(waveform, **parameters)
    if operation in {"hard_clip", "clipping"}:
        return clip(waveform, soft=False, **parameters)
    if operation in {"soft_clip", "tanh_distortion"}:
        return clip(waveform, soft=True, **parameters)
    if operation == "polarity_inversion":
        return -ensure_stereo(waveform)
    if operation in {"fade", "gain_transition"}:
        return fade(waveform, sample_rate, **parameters)
    if operation == "trim":
        return trim_seconds(waveform, sample_rate, **parameters)
    if operation == "padding":
        return zero_pad(waveform, sample_rate, **parameters)
    if operation in {"dropout", "time_mask"}:
        return dropout(waveform, sample_rate, seed=seed, **parameters)
    if operation in {"white_noise", "gaussian_noise", "short_noise"}:
        return add_white_noise(waveform, seed=seed, **parameters)
    if operation in {"pink_noise", "color_noise"}:
        return add_pink_noise(waveform, seed=seed, **parameters)
    if operation == "pink_noise":
        return add_pink_noise(waveform, seed=seed, **parameters)
    if operation == "environmental_noise":
        if "noise" not in resources:
            raise ValueError("environmental_noise requires resources['noise']")
        return add_environmental_noise(
            waveform, resources["noise"], sample_rate, seed=seed, **parameters
        )
    if operation in {"rir", "room_simulator"}:
        if "rir" not in resources:
            raise ValueError("rir requires resources['rir']")
        return add_rir(waveform, resources["rir"], **parameters)
    if operation == "echo":
        return add_echo(waveform, sample_rate, **parameters)
    if operation in {"eq", "parametric_eq", "seven_band_eq", "peaking"}:
        return equalizer(waveform, sample_rate, **parameters)
    if operation in {"lowpass", "air_absorption"}:
        return lowpass(waveform, sample_rate, parameters.get("cutoff_hz", 12000.0))
    if operation == "highpass":
        return highpass(waveform, sample_rate, parameters.get("cutoff_hz", 60.0))
    if operation == "bandpass":
        return bandpass(
            waveform, sample_rate, parameters.get("low_hz", 80.0),
            parameters.get("high_hz", 16000.0),
        )
    if operation == "bandstop":
        return bandstop(
            waveform, sample_rate, parameters.get("low_hz", 2000.0),
            parameters.get("high_hz", 4000.0),
        )
    if operation in {"shift", "temporal_shift"}:
        return temporal_shift(waveform, sample_rate, **parameters)
    if operation == "pitch_shift":
        return pitch_shift(waveform, sample_rate, **parameters)
    if operation == "time_stretch":
        return time_stretch(waveform, sample_rate, **parameters)
    if operation in {"resample_roundtrip", "aliasing"}:
        return resample_roundtrip(
            waveform,
            sample_rate,
            parameters.get("intermediate_sample_rate", 24000),
        )
    if operation in {"quantization", "bit_crush"}:
        return quantize(waveform, seed=seed, **parameters)
    if operation in {"codec", "mp3", "aac", "opus"}:
        codec = parameters.pop("codec", operation if operation != "codec" else "mp3")
        return codec_roundtrip(
            waveform,
            sample_rate,
            codec=codec,
            **parameters,
        )
    if operation == "encodec":
        weights = parameters.pop("weights", resources.get("encodec_weights"))
        if not weights:
            raise ValueError("encodec requires weights or resources['encodec_weights']")
        return encodec_roundtrip(waveform, sample_rate, weights=str(weights), **parameters)
    if operation == "dac":
        weights = parameters.pop("weights", resources.get("dac_weights"))
        if not weights:
            raise ValueError("dac requires weights or resources['dac_weights']")
        return dac_roundtrip(waveform, sample_rate, weights=str(weights), **parameters)
    if operation == "packet_loss":
        return packet_loss(waveform, sample_rate, seed=seed, **parameters)
    if operation == "mastering":
        output = equalizer(waveform, sample_rate, kind="low_shelf", frequency_hz=120.0, gain_db=parameters.get("eq_low_db", 1.0))
        output = compress(
            output,
            sample_rate,
            threshold_db=parameters.get("threshold_db", -18.0),
            ratio=parameters.get("ratio", 2.0),
            attack_ms=parameters.get("attack_ms", 20.0),
            release_ms=parameters.get("release_ms", 100.0),
        )
        return limit(output, drive_db=parameters.get("drive_db", 3.0))
    raise KeyError(f"Unknown corruption operation: {operation}")


def default_operation_specs() -> tuple[OperationSpec, ...]:
    """Return the paper-aligned single-operation pool used by Audio RA.

    The ICLR supplementary implementation samples one operation uniformly and
    samples that operation's physical parameters from its own range.
    Pitch shift follows the SOFIA/legacy training range of -2 to +2 semitones.
    Compression and Opus are active project extensions. EnCodec and DAC keep
    their implementations below but are deferred from both training and fixed
    evaluation. AAC remains a fixed-evaluation condition; MP3 is omitted
    because the benchmark source collection is already MP3.
    """

    return (
        OperationSpec("gain", "A", "gain_db", -12.0, 12.0),
        OperationSpec("compression", "A", "ratio", 2.0, 6.0),
        OperationSpec("limiter", "A", "drive_db", 0.0, 6.0),
        OperationSpec("clipping", "A", "drive_db", 0.0, 6.0),
        OperationSpec("dropout", "A", "missing_ratio", 0.001, 0.01),
        OperationSpec("white_noise", "B", "snr_db_value", 10.0, 30.0),
        OperationSpec("pink_noise", "B", "snr_db_value", 10.0, 30.0),
        OperationSpec("environmental_noise", "B", "snr_db_value", 10.0, 30.0, resource_key="noise"),
        OperationSpec("rir", "B", "dry_wet", 0.2, 0.5, resource_key="rir"),
        OperationSpec("eq", "C", "gain_db", -6.0, 6.0),
        OperationSpec("lowpass", "C", "cutoff_hz", 8000.0, 16000.0),
        OperationSpec("highpass", "C", "cutoff_hz", 30.0, 120.0),
        OperationSpec("bandpass", "C", "low_hz", 80.0, 2000.0),
        OperationSpec("bandstop", "C", "low_hz", 2000.0, 6000.0),
        OperationSpec("polarity_inversion", "C", "unused", 0.0, 1.0),
        OperationSpec("tanh_distortion", "C", "drive_db", 0.0, 6.0),
        OperationSpec("temporal_shift", "D", "shift_fraction", -0.5, 0.5),
        OperationSpec("pitch_shift", "D", "semitones", -2.0, 2.0),
        OperationSpec("time_stretch", "D", "speed_factor", 0.6, 1.5, False),
        OperationSpec("resample_roundtrip", "E", "intermediate_sample_rate", 8000.0, 16000.0),
        OperationSpec("quantization", "E", "bits", 8.0, 8.0),
        OperationSpec("codec", "E", "bitrate_kbps", 64.0, 128.0),
    )


def _sample_spec_value(spec: OperationSpec, rng: random.Random) -> float:
    """Sample uniformly from the operation-specific physical range."""

    return rng.uniform(spec.minimum, spec.maximum)


class AudioRandAugment:
    """RandAugment adapted to heterogeneous music operation units.

    ``num_layers`` and ``apply_probability`` are the only policy-level
    hyperparameters. Physical ranges live in ``OperationSpec`` and are sampled
    independently, so Hz is never treated as if it were dB or semitones.
    """

    def __init__(
        self,
        *,
        num_layers: int = 1,
        magnitude: Optional[float] = None,
        apply_probability: float = 0.9,
        specs: Iterable[OperationSpec] = (),
        family_balanced: bool = False,
        allow_length_change: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.num_layers = max(int(num_layers), 0)
        # Backwards-compatible argument; paper RA samples per-operation ranges.
        self.magnitude = magnitude
        self.apply_probability = float(apply_probability)
        if not 0.0 <= self.apply_probability <= 1.0:
            raise ValueError("apply_probability must be in [0, 1]")
        self.specs = tuple(specs) or default_operation_specs()
        self.family_balanced = bool(family_balanced)
        self.allow_length_change = bool(allow_length_change)
        self.seed = seed

    def _select(
        self, rng: random.Random, used: set[str], available_resources: set[str]
    ) -> OperationSpec:
        candidates = [
            spec
            for spec in self.specs
            if spec.operation not in used
            and (self.allow_length_change or spec.length_preserving)
            and (spec.resource_key is None or spec.resource_key in available_resources)
        ]
        if not candidates:
            candidates = [
                spec
                for spec in self.specs
                if (self.allow_length_change or spec.length_preserving)
                and (spec.resource_key is None or spec.resource_key in available_resources)
            ]
        if self.family_balanced:
            families = sorted({spec.family for spec in candidates})
            family = rng.choice(families)
            candidates = [spec for spec in candidates if spec.family == family]
        return rng.choice(candidates)

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        seed: Optional[int] = None,
        resources: Optional[Mapping[str, Any]] = None,
        trace_context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        rng = random.Random(self.seed if seed is None else seed)
        output = ensure_stereo(waveform)
        applied: list[dict[str, Any]] = []
        used: set[str] = set()
        for order in range(self.num_layers):
            spec = self._select(rng, used, set((resources or {}).keys()))
            used.add(spec.operation)
            if rng.random() > self.apply_probability:
                applied.append(
                    {
                        "order": order,
                        "operation": spec.operation,
                        "family": spec.family,
                        "applied": False,
                    }
                )
                continue
            value = _sample_spec_value(spec, rng)
            params: dict[str, Any] = {spec.parameter: value}
            if spec.operation == "fade":
                params.update(direction=rng.choice(["in", "out"]), shape="half_sine")
            elif spec.operation == "dropout":
                params["frame_ms"] = rng.uniform(20.0, 40.0)
            elif spec.operation == "eq":
                params.update(kind=rng.choice(["bell", "low_shelf", "high_shelf"]), frequency_hz=rng.choice([120.0, 3000.0, 8000.0]), q=1.0)
            elif spec.operation == "bandpass":
                params = {"low_hz": value, "high_hz": rng.choice([8000.0, 12000.0, 16000.0])}
            elif spec.operation == "bandstop":
                params = {"low_hz": value, "high_hz": min(value + rng.uniform(1000.0, 4000.0), 16000.0)}
            elif spec.operation == "polarity_inversion":
                params = {}
            elif spec.operation == "resample_roundtrip":
                params["intermediate_sample_rate"] = int(round(value / 1000.0) * 1000)
            elif spec.operation == "quantization":
                params["bits"] = int(round(value))
            elif spec.operation == "compression":
                params.update(threshold_db=-18.0, attack_ms=10.0, release_ms=100.0)
            elif spec.operation == "codec":
                params.update(codec="opus", bitrate_kbps=int(round(value)))
            elif spec.operation == "encodec":
                # The official 48 kHz model accepts categorical bandwidths,
                # not arbitrary values inside a numeric interval.
                params = {
                    "bandwidth_kbps": rng.choice([3.0, 6.0, 12.0]),
                    "device": str((resources or {}).get("codec_device", "cpu")),
                }
            elif spec.operation == "dac":
                params = {
                    "n_quantizers": None,
                    "device": str((resources or {}).get("codec_device", "cpu")),
                }
            selected_resources = dict(resources or {})
            if spec.resource_key is not None:
                resource = selected_resources[spec.resource_key]
                if callable(resource):
                    resource = resource()
                selected_resources = {spec.resource_key: resource}
            operation_started = time.perf_counter()
            if spec.operation in {"codec", "encodec", "dac"}:
                print(
                    "randaugment_operation_start "
                    f"operation={spec.operation} "
                    f"track_id={(trace_context or {}).get('track_id', '')} "
                    f"epoch={(trace_context or {}).get('epoch', '')} "
                    f"seed={(trace_context or {}).get('seed', seed)} "
                    f"samples={output.shape[-1]} parameters={params}",
                    flush=True,
                )
            output = apply_operation(
                output,
                sample_rate,
                spec.operation,
                params,
                resources=selected_resources,
                seed=rng.randrange(1 << 62),
            )
            operation_seconds = time.perf_counter() - operation_started
            if spec.operation in {"codec", "encodec", "dac"} or operation_seconds >= 10.0:
                print(
                    "randaugment_operation_end "
                    f"operation={spec.operation} "
                    f"track_id={(trace_context or {}).get('track_id', '')} "
                    f"epoch={(trace_context or {}).get('epoch', '')} "
                    f"seconds={operation_seconds:.3f}",
                    flush=True,
                )
            applied.append(
                {
                    "order": order,
                    "operation": spec.operation,
                    "family": spec.family,
                    "parameters": params,
                    "applied": True,
                }
            )
        return output, applied


def core_condition_registry() -> list[dict[str, Any]]:
    """Return fixed benchmark conditions without model-dependent choices."""

    return [
        {"condition_id": "op-a01-gain-m03", "family": "A", "operation": "gain", "parameters": {"gain_db": -3.0}},
        {"condition_id": "op-a01-gain-m06", "family": "A", "operation": "gain", "parameters": {"gain_db": -6.0}},
        {"condition_id": "op-a01-gain-m12", "family": "A", "operation": "gain", "parameters": {"gain_db": -12.0}},
        {"condition_id": "op-a01-peak-m03", "family": "A", "operation": "peak_normalization", "parameters": {"target_dbfs": -3.0}},
        {"condition_id": "op-a01-loudness-m16", "family": "A", "operation": "loudness_normalization", "parameters": {"target_lufs": -16.0}},
        {"condition_id": "op-a02-compression-medium", "family": "A", "operation": "compression", "parameters": {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}},
        {"condition_id": "op-a03-fade-in-1s", "family": "A", "operation": "fade", "parameters": {"direction": "in", "seconds": 1.0, "shape": "half_sine"}},
        {"condition_id": "op-a04-dropout-p05", "family": "A", "operation": "dropout", "parameters": {"missing_ratio": 0.005, "frame_ms": 30.0}},
        {"condition_id": "op-b01-white-snr20", "family": "B", "operation": "white_noise", "parameters": {"snr_db_value": 20.0}},
        {"condition_id": "op-b02-echo-300ms", "family": "B", "operation": "echo", "parameters": {"delay_ms": 300.0, "decay": 0.35}},
        {"condition_id": "op-c01-eq-high-p03", "family": "C", "operation": "eq", "parameters": {"kind": "high_shelf", "frequency_hz": 8000.0, "gain_db": 3.0, "q": 1.0}},
        {"condition_id": "op-c02-lowpass-12k", "family": "C", "operation": "lowpass", "parameters": {"cutoff_hz": 12000.0}},
        {"condition_id": "op-c03-highpass-60", "family": "C", "operation": "highpass", "parameters": {"cutoff_hz": 60.0}},
        {"condition_id": "op-d01-pitch-m1", "family": "D", "operation": "pitch_shift", "parameters": {"semitones": -1.0}},
        {"condition_id": "op-d01-pitch-p1", "family": "D", "operation": "pitch_shift", "parameters": {"semitones": 1.0}},
        {"condition_id": "op-d02-time-slow", "family": "D", "operation": "time_stretch", "parameters": {"speed_factor": 0.9}},
        {"condition_id": "op-e01-resample-24k", "family": "E", "operation": "resample_roundtrip", "parameters": {"intermediate_sample_rate": 24000}},
        {"condition_id": "op-e02-quantize-8bit", "family": "E", "operation": "quantization", "parameters": {"bits": 8, "dither": False}},
        {"condition_id": "op-e03-aac-128", "family": "E", "operation": "codec", "parameters": {"codec": "aac", "bitrate_kbps": 128}},
        {"condition_id": "op-e04-aac-128", "family": "E", "operation": "codec", "parameters": {"codec": "aac", "bitrate_kbps": 128}},
        {"condition_id": "op-e05-opus-96", "family": "E", "operation": "codec", "parameters": {"codec": "opus", "bitrate_kbps": 96}},
        {"condition_id": "op-e06-packet-loss-p05", "family": "E", "operation": "packet_loss", "parameters": {"frame_ms": 20.0, "loss_rate": 0.05, "mode": "hold", "burst": False}},
    ]
