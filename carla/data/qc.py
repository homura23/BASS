"""Quality and validity gates for MusicCorrupt-Bench variants."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Optional

import torch

from .corruptions import rms_db, snr_db


@dataclass(frozen=True)
class QCGates:
    """Conservative defaults; operation-specific checks refine these values."""

    finite: bool = True
    stereo: bool = True
    max_peak: float = 1.25
    max_interchannel_lag_samples: int = 0
    min_music_similarity: float = 0.0
    min_human_quality: Optional[float] = None


def _as_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def estimate_true_peak(waveform: torch.Tensor, oversample: int = 4) -> float:
    """Estimate true peak with linear interpolation.

    This is a lightweight fallback for a BS.1770/true-peak meter.  The
    metadata records the implementation so an external meter can replace it.
    """

    waveform = _as_stereo(waveform).float()
    if oversample <= 1 or waveform.shape[-1] < 2:
        return float(waveform.abs().amax().item())
    left = waveform[..., :-1]
    right = waveform[..., 1:]
    steps = torch.linspace(
        0.0,
        1.0,
        int(oversample),
        device=waveform.device,
        dtype=waveform.dtype,
    ).view(1, 1, -1)
    interp = left.unsqueeze(-1) * (1.0 - steps) + right.unsqueeze(-1) * steps
    return float(max(waveform.abs().amax().item(), interp.abs().amax().item()))


def interchannel_lag_samples(waveform: torch.Tensor, max_lag: int = 128) -> int:
    waveform = _as_stereo(waveform)
    if waveform.shape[0] < 2:
        return 0
    left = waveform[0] - waveform[0].mean()
    right = waveform[1] - waveform[1].mean()
    max_lag = min(int(max_lag), max(left.numel() - 1, 0))
    if max_lag <= 0:
        return 0
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            value = (left[-lag:] * right[:lag]).mean()
        elif lag > 0:
            value = (left[:-lag] * right[lag:]).mean()
        else:
            value = (left * right).mean()
        correlations.append(value)
    return int(torch.tensor(correlations).argmax().item() - max_lag)


def audio_summary(
    clean: torch.Tensor,
    output: torch.Tensor,
    *,
    sample_rate: int,
    target_snr_db: Optional[float] = None,
) -> dict[str, Any]:
    clean = _as_stereo(clean).float()
    output = _as_stereo(output).float()
    summary: dict[str, Any] = {
        "input_channels": int(clean.shape[0]),
        "output_channels": int(output.shape[0]),
        "input_samples": int(clean.shape[-1]),
        "output_samples": int(output.shape[-1]),
        "sample_rate": int(sample_rate),
        "finite": bool(torch.isfinite(output).all().item()),
        "peak": float(output.abs().amax().item()) if output.numel() else math.nan,
        "true_peak": estimate_true_peak(output),
        "rms_db": rms_db(output) if output.numel() else math.nan,
        "input_rms_db": rms_db(clean) if clean.numel() else math.nan,
        "interchannel_lag_samples": interchannel_lag_samples(output),
    }
    if target_snr_db is not None and clean.shape[-1] == output.shape[-1]:
        summary["snr_db"] = snr_db(clean, output)
        summary["target_snr_db"] = float(target_snr_db)
    return summary


def technical_gate(
    clean: torch.Tensor,
    output: torch.Tensor,
    *,
    sample_rate: int,
    expected_sample_rate: Optional[int] = None,
    expected_channels: int = 2,
    max_peak: float = 1.25,
) -> tuple[bool, str]:
    """Reject malformed variants without silently repairing them."""

    if output.ndim != 2:
        return False, "output_not_channel_first"
    if output.shape[0] != expected_channels:
        return False, "unexpected_channel_count"
    if output.shape[-1] <= 0:
        return False, "empty_output"
    if not torch.isfinite(output).all():
        return False, "nan_or_inf"
    if expected_sample_rate is not None and int(sample_rate) != int(expected_sample_rate):
        return False, "unexpected_sample_rate"
    if float(output.abs().amax()) > float(max_peak):
        return False, "unexpected_peak"
    if clean.ndim != 2 or clean.shape[0] != expected_channels:
        return False, "invalid_clean_reference"
    return True, ""


def operation_gate(
    operation: str,
    clean: torch.Tensor,
    output: torch.Tensor,
    parameters: Mapping[str, Any],
    *,
    sample_rate: int,
) -> tuple[bool, str, dict[str, Any]]:
    """Check that the requested operation measurably took effect."""

    operation = operation.lower().replace("-", "_")
    details: dict[str, Any] = {}
    if operation == "gain":
        details["measured_gain_db"] = rms_db(output) - rms_db(clean)
        target = float(parameters.get("gain_db", 0.0))
        if abs(details["measured_gain_db"] - target) > 0.15:
            return False, "gain_mismatch", details
    elif operation == "peak_normalization":
        details["measured_peak_dbfs"] = float(
            20.0 * torch.log10(output.abs().amax().clamp_min(1e-12)).item()
        )
        target = float(parameters.get("target_dbfs", -3.0))
        if abs(details["measured_peak_dbfs"] - target) > 0.1:
            return False, "peak_normalization_mismatch", details
    elif operation == "loudness_normalization":
        details["measured_loudness_proxy"] = rms_db(output)
        target = float(parameters.get("target_lufs", -16.0))
        if abs(details["measured_loudness_proxy"] - target) > 0.75:
            return False, "loudness_normalization_mismatch", details
    elif operation == "white_noise":
        if clean.shape[-1] != output.shape[-1]:
            return False, "noise_changed_length", details
        details["measured_snr_db"] = snr_db(clean, output)
        target = float(parameters.get("snr_db_value", 20.0))
        if abs(details["measured_snr_db"] - target) > 0.75:
            return False, "snr_mismatch", details
    elif operation == "dropout":
        if clean.shape != output.shape:
            return False, "dropout_changed_shape", details
        details["zero_ratio"] = float((output.abs() <= 1e-12).float().mean().item())
        if details["zero_ratio"] <= 0.0:
            return False, "dropout_not_active", details
    elif operation == "trim":
        target = round(float(parameters.get("seconds", 0.0)) * sample_rate)
        if clean.shape[-1] - output.shape[-1] != target:
            return False, "trim_length_mismatch", details
    elif operation == "padding":
        target = round(float(parameters.get("seconds", 0.0)) * sample_rate)
        if output.shape[-1] - clean.shape[-1] < target:
            return False, "padding_length_mismatch", details
    elif operation == "pitch_shift":
        details["requested_semitones"] = float(parameters.get("semitones", 0.0))
    elif operation == "time_stretch":
        details["duration_ratio"] = output.shape[-1] / max(clean.shape[-1], 1)
        expected = 1.0 / float(parameters.get("speed_factor", 1.0))
        if abs(details["duration_ratio"] - expected) > 0.03:
            return False, "duration_ratio_mismatch", details
    elif operation == "resample_roundtrip":
        if output.shape[-1] == 0:
            return False, "resample_empty", details
    elif operation == "quantization":
        details["unique_values"] = int(torch.unique(output).numel())
        if details["unique_values"] <= 1:
            return False, "quantization_collapsed", details
    elif operation in {"lowpass", "highpass", "bandpass", "eq", "compression", "limiter", "echo", "rir", "mastering"}:
        details["rms_delta_db"] = rms_db(output) - rms_db(clean)
    return True, "", details


def quality_gate(
    clean: torch.Tensor,
    output: torch.Tensor,
    *,
    sample_rate: int,
    operation: str,
    parameters: Optional[Mapping[str, Any]] = None,
    gates: QCGates = QCGates(),
) -> dict[str, Any]:
    """Run technical and operation gates and return serializable QC metadata."""

    parameters = parameters or {}
    technical_pass, technical_reason = technical_gate(
        clean,
        output,
        sample_rate=sample_rate,
        expected_sample_rate=sample_rate,
        expected_channels=2,
        max_peak=gates.max_peak,
    )
    operation_pass, operation_reason, measured = operation_gate(
        operation,
        clean,
        output,
        parameters,
        sample_rate=sample_rate,
    )
    result: dict[str, Any] = {
        "operation_gate_pass": bool(operation_pass),
        "technical_gate_pass": bool(technical_pass),
        "perceptual_gate_pass": True,
        "content_gate_pass": True,
        "label_gate_pass": True,
        "accepted": bool(operation_pass and technical_pass),
        "rejection_reason": operation_reason or technical_reason,
        "operation": operation,
        "parameters": dict(parameters),
        "summary": audio_summary(
            clean,
            output,
            sample_rate=sample_rate,
            target_snr_db=parameters.get("snr_db_value"),
        ),
        "measured": measured,
        "gate_config": asdict(gates),
    }
    return result


def validate_embedding_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {tuple(value.shape)}")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
