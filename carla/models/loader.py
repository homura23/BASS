"""Builders and checkpoint utilities for CARLA R0/R1."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from .flow import VectorGlow
from .model import CarlaModel
from .robustness import FusionWrapper
from ..utils.sofia_bridge import build_fusion
from ..utils.run_io import torch_save_atomic


def build_flow(cfg: Dict, device: torch.device) -> VectorGlow:
    flow = cfg["flow"]
    return VectorGlow(
        dim=int(flow["dim"]),
        K=int(flow["K"]),
        hidden=int(flow["hidden"]),
        R=float(flow["R"]),
        permutation=str(flow.get("permutation", "reverse")),
    ).to(device)


def build_model(cfg: Dict, device: torch.device, variant: str | None = None) -> CarlaModel:
    fusion = build_fusion(cfg, device)
    version = variant or str(cfg.get("model_version", "r0"))
    wrapped = FusionWrapper(fusion, version=version)
    return CarlaModel(wrapped, build_flow(cfg, device)).to(device)


def validate_base_feature_contract(
    checkpoint_cfg: Mapping[str, Any] | None, requested_cfg: Mapping[str, Any]
) -> None:
    if not checkpoint_cfg:
        return
    for key in ("encoder_compatibility", "audio_branches", "fusion"):
        checkpoint_value = checkpoint_cfg.get(key)
        requested_value = requested_cfg.get(key)
        if checkpoint_value != requested_value:
            raise ValueError(
                f"Base checkpoint feature contract mismatch for {key}: "
                f"checkpoint={checkpoint_value!r} requested={requested_value!r}"
            )


def load_base_v1(
    checkpoint: str, cfg: Dict, device: torch.device, variant: str | None = None
) -> CarlaModel:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("arch", "v1") != "v1":
        raise ValueError(f"CARLA currently builds on v1 only; checkpoint arch={state.get('arch')!r}")
    validate_base_feature_contract(state.get("config"), cfg)
    model = build_model(cfg, device, variant or cfg.get("model_version", "r0"))
    model.fusion.load_state_dict(state["fusion"], strict=True)
    model.flow.load_state_dict(state["flow"], strict=True)
    return model


def save_carla_checkpoint(
    path: str, model: CarlaModel, cfg: Dict, *, stage: str, epoch: int = 0,
    metadata: Dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: Dict[str, Any] | None = None,
) -> None:
    torch_save_atomic(path, {
        "project": "CARLA",
        "arch": "v1",
        "model_version": model.robust_fusion.variant,
        "robust_variant": model.robust_fusion.variant,
        "fusion": model.fusion.state_dict(),
        "flow": model.flow.state_dict(),
        "config": cfg,
        "stage": stage,
        "epoch": epoch,
        "metadata": dict(metadata or {}),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "training_state": dict(training_state or {}),
    })


def load_carla_checkpoint(checkpoint: str, device: torch.device) -> Tuple[CarlaModel, Dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    project = state.get("project")
    if project != "CARLA":
        raise ValueError("Checkpoint is not a CARLA artifact")
    cfg = state["config"]
    version = state.get("model_version", state.get("robust_variant", cfg.get("model_version", "r0")))
    model = build_model(cfg, device, version)
    model.fusion.load_state_dict(state["fusion"], strict=True)
    model.flow.load_state_dict(state["flow"], strict=True)
    return model, cfg
