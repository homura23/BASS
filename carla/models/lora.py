"""Pluggable generator LoRA; registered after robust hooks (R(h) + LoRA(h))."""
from __future__ import annotations

import math
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


class LoRAPair(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        self.scale = alpha / rank
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.b(self.a(self.dropout(x)))


def inject_lora(
    module: nn.Module,
    targets: Sequence[str],
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Dict:
    remove_lora(module)
    lora = nn.ModuleDict()
    handles: List[object] = []
    matched = []
    for name, layer in list(module.named_modules()):
        if isinstance(layer, nn.Linear) and any(target in name for target in targets):
            key = name.replace(".", "__")
            lora[key] = LoRAPair(
                layer.in_features, layer.out_features, rank, alpha, dropout
            )
            matched.append(name)

            def make_hook(adapter_key: str):
                def hook(_layer, inputs, output):
                    return output + module.lora[adapter_key](inputs[0])
                return hook

            handles.append(layer.register_forward_hook(make_hook(key)))
    if not matched:
        raise ValueError(f"No Linear layer matched LoRA targets {list(targets)}")
    module.lora = lora.to(next(module.parameters()).device)
    module._lora_handles = handles
    return {
        "targets": list(targets),
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "matched": matched,
    }


def remove_lora(module: nn.Module) -> None:
    for handle in getattr(module, "_lora_handles", []):
        handle.remove()
    module._lora_handles = []
    if "lora" in getattr(module, "_modules", {}):
        module._modules.pop("lora")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value for name, value in module.state_dict().items() if "lora" in name}


def load_lora_checkpoint(
    model: nn.Module, checkpoint: str | Path | Dict[str, Any], *,
    base_checkpoint: str | Path, strict: bool = True,
) -> Dict[str, Any]:
    """Inject and strictly load one LoRA bound to an exact base checkpoint."""

    state = torch.load(checkpoint, map_location="cpu") if not isinstance(checkpoint, dict) else checkpoint
    if state.get("project") != "CARLA":
        raise ValueError("LoRA checkpoint is not a CARLA artifact")
    expected_hash = state.get("base_sha256")
    actual_hash = checkpoint_sha256(base_checkpoint)
    if strict and not expected_hash:
        raise ValueError("LoRA checkpoint has no base_sha256; legacy permissive loading is forbidden")
    if expected_hash and str(expected_hash) != actual_hash:
        raise ValueError(f"LoRA/base mismatch: expected {expected_hash}, got {actual_hash}")
    if state.get("robust_variant") != model.robust_fusion.variant:
        raise ValueError("LoRA robust_variant does not match the loaded base")
    cfg = state["lora_cfg"]
    inject_lora(model.fusion, cfg["targets"], rank=int(cfg["rank"]),
                alpha=float(cfg["alpha"]), dropout=float(cfg.get("dropout", 0.0)))
    expected = set(lora_state_dict(model.fusion))
    actual = set(state.get("lora", {}))
    if expected != actual:
        raise ValueError(f"Invalid LoRA keys: missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}")
    full = model.fusion.state_dict()
    full.update(state["lora"])
    model.fusion.load_state_dict(full, strict=True)
    return state
