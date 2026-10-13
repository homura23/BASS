"""R0/R1 fusion wrapper.

CARLA exposes two release versions: R0 is the clean baseline and R1 uses
the same model with real-only RandAugment during training. The versions
differ by training protocol, not by an additional canonicalizer topology.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FusionWrapper(nn.Module):
    """Expose the common fusion interface used by training and LoRA."""

    VALID_VERSIONS = {"r0", "r1"}

    def __init__(self, fusion: nn.Module, version: str = "r0") -> None:
        super().__init__()
        version = str(version).lower()
        if version not in self.VALID_VERSIONS:
            raise ValueError(f"Unknown CARLA model version {version!r}; use r0 or r1")
        self.fusion = fusion
        self.variant = version

    def forward(self, embeddings: Dict[str, torch.Tensor], return_details: bool = False):
        fused = self.fusion(embeddings)
        if return_details:
            return {
                "fused": fused,
                "base_fused": fused,
                "branch_base": {},
                "branch_corrected": {},
            }
        return fused

    def forward_base(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion(embeddings)

    def robustness_parameters(self):
        return iter(())

    def close(self) -> None:
        return None
