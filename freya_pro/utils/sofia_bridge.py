"""Project-relative bridge to the official SOFIA runtime package.

CARLA does not redistribute SOFIA source or pretrained weights. Set
``SOFIA_ROOT`` to a checkout of https://github.com/homura23/SOFIA, or place a
sibling checkout next to CARLA. Weight paths in CARLA configs are relative to
that SOFIA checkout, matching the official SOFIA README.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn


def default_sofia_root() -> Path:
    configured = os.environ.get("SOFIA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # <code>/CARLA/freya_pro/utils/sofia_bridge.py -> <code>/sofia
    return Path(__file__).resolve().parents[2].parent / "sofia"


SOFIA_ROOT = default_sofia_root()
if not SOFIA_ROOT.is_dir():
    raise ModuleNotFoundError(
        "SOFIA is not available. Clone https://github.com/homura23/SOFIA "
        "and set SOFIA_ROOT to that checkout."
    )
if str(SOFIA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SOFIA_ROOT.parent))

from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate  # noqa: E402
from sofia.encoders.fxpp import FxPPEncoder  # noqa: E402
from sofia.encoders.mert import MertEncoder  # noqa: E402
from sofia.encoders.muq import MuQEncoder  # noqa: E402
from sofia.encoders.wave import Wave2Vec2Encoder  # noqa: E402
from sofia.models.fusion import MultiBranchFusion  # noqa: E402


def _path_from_sofia(value: str | None, sofia_root: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((sofia_root / path).resolve())


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def encoder_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def encoder_device_map(branches: Dict, primary: torch.device) -> Dict[str, torch.device]:
    names = [name for name in ("fxpp", "muq", "mert", "wave") if branches.get(name, {}).get("enabled")]
    count = torch.cuda.device_count() if primary.type == "cuda" else 0
    if primary.type == "cuda" and primary.index is not None:
        return {name: primary for name in names}
    if count <= 1:
        return {name: primary for name in names}
    return {name: torch.device(f"cuda:{index % count}") for index, name in enumerate(names)}


def _repair_legacy_positional_weight_norm(encoder: nn.Module, checkpoint_dir: str) -> None:
    checkpoint = Path(checkpoint_dir) / "pytorch_model.bin"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parameters = dict(encoder.model.named_parameters())
    repaired = []
    for old_name, new_suffix in (("weight_g", "parametrizations.weight.original0"),
                                 ("weight_v", "parametrizations.weight.original1")):
        matches = [key for key in state if key.endswith(f"pos_conv_embed.conv.{old_name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one legacy {old_name} key in {checkpoint}, got {matches}")
        target_matches = [key for key in parameters if key.endswith(f"pos_conv_embed.conv.{new_suffix}")]
        if len(target_matches) != 1:
            raise KeyError(f"Expected one migrated positional-conv parameter, got {target_matches}")
        target_name = target_matches[0]
        source = state[matches[0]]
        target = parameters[target_name]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(f"Shape mismatch for {target_name}: {tuple(source.shape)} vs {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        repaired.append(target_name)
    encoder.legacy_weight_norm_repaired = tuple(repaired)


def build_encoders(
    branches: Dict,
    primary: torch.device,
    compatibility: Mapping | None = None,
) -> Dict[str, nn.Module]:
    compatibility = dict(compatibility or {})
    mode = str(compatibility.get("mode", "sofia_legacy"))
    if mode not in {"sofia_legacy", "checkpoint_repaired"}:
        raise ValueError(f"Unknown encoder compatibility mode: {mode}")
    initialization_seed = int(compatibility.get("initialization_seed", 42))
    sofia_root = Path(compatibility.get("sofia_root") or SOFIA_ROOT).expanduser().resolve()
    devices = encoder_device_map(branches, primary)
    output: Dict[str, nn.Module] = {}
    cuda_devices = sorted({device.index for device in devices.values()
                           if device.type == "cuda" and device.index is not None})
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initialization_seed)
        if branches.get("fxpp", {}).get("enabled"):
            cfg = branches["fxpp"]
            output["fxpp"] = FxPPEncoder(
                weights=_path_from_sofia(cfg.get("weights"), sofia_root),
                repo_path=_path_from_sofia(cfg.get("repo_path"), sofia_root),
                device=devices["fxpp"],
            )
        if branches.get("muq", {}).get("enabled"):
            cfg = branches["muq"]
            output["muq"] = MuQEncoder(
                weights=_path_from_sofia(cfg.get("weights"), sofia_root),
                repo_path=_path_from_sofia(cfg.get("repo_path"), sofia_root),
                device=devices["muq"],
            )
        if branches.get("mert", {}).get("enabled"):
            cfg = branches["mert"]
            weights = _path_from_sofia(cfg.get("weights"), sofia_root)
            output["mert"] = MertEncoder(
                weights=weights, device=devices["mert"],
                layer_pooling=str(cfg.get("layer_pooling", "layer_mean")),
            )
            if mode == "checkpoint_repaired":
                _repair_legacy_positional_weight_norm(output["mert"], str(weights))
        if branches.get("wave", {}).get("enabled"):
            cfg = branches["wave"]
            weights = _path_from_sofia(cfg.get("weights"), sofia_root)
            output["wave"] = Wave2Vec2Encoder(
                weights=weights, device=devices["wave"],
                layer_pooling=str(cfg.get("layer_pooling", "layer_mean")),
            )
            if mode == "checkpoint_repaired":
                _repair_legacy_positional_weight_norm(output["wave"], str(weights))
    if not output:
        raise ValueError("No SOFIA audio branch is enabled")
    for encoder in output.values():
        encoder.encoder_compatibility_mode = mode
        encoder.encoder_initialization_seed = initialization_seed
        set_requires_grad(encoder, False)
        encoder.eval()
    return output


def encode_branches(
    encoders: Dict[str, nn.Module],
    audio: Dict[str, torch.Tensor] | torch.Tensor,
    *,
    sample_rate: int | Dict[str, int],
    device: torch.device,
    differentiable: bool = False,
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for name, encoder in encoders.items():
        branch_audio = audio[name] if isinstance(audio, dict) else audio
        branch_sr = sample_rate[name] if isinstance(sample_rate, dict) else sample_rate
        branch_device = encoder_device(encoder)
        branch_audio = branch_audio.to(branch_device, non_blocking=True)
        if differentiable:
            method = getattr(encoder, "encode_differentiable", None)
            if method is None:
                raise RuntimeError(f"Encoder {name!r} has no differentiable attack interface")
            value = method(branch_audio, sample_rate=int(branch_sr))
        else:
            value = encoder(branch_audio, sample_rate=int(branch_sr))
        output[name] = value.to(device)
    return output


def build_fusion(cfg: Dict, device: torch.device) -> MultiBranchFusion:
    branches = {name: value for name, value in cfg["audio_branches"].items() if value.get("enabled")}
    dims = {name: int(value["embed_dim"]) for name, value in branches.items()}
    fusion = cfg["fusion"]
    return MultiBranchFusion(
        branch_input_dims=dims,
        projector_dim=int(fusion["projector_dim"]),
        fusion_hidden=fusion.get("fusion_hidden", [512]),
        fusion_out=int(fusion["fusion_out"]),
        dropout=float(fusion.get("dropout", 0.1)),
        l2_norm=False,
        mode=str(fusion.get("mode", "moe")),
        moe_routing=str(fusion.get("moe_routing", "dense")),
        top_k=int(fusion.get("top_k", 1)),
    ).to(device)
