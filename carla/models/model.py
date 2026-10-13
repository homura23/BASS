"""CARLA density model used by the R0 and R1 release variants."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .flow import VectorGlow
from .robustness import FusionWrapper


class CarlaModel(nn.Module):
    """Compose external encoder fusion with the CARLA density flow."""

    def __init__(self, fusion: FusionWrapper, flow: VectorGlow) -> None:
        super().__init__()
        self.robust_fusion = fusion
        self.flow = flow

    @property
    def fusion(self) -> nn.Module:
        return self.robust_fusion.fusion

    def canonicalize(self, embeddings: Dict[str, torch.Tensor], return_details: bool = False):
        return self.robust_fusion(embeddings, return_details=return_details)

    def base_fuse(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.robust_fusion.forward_base(embeddings)

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        fused = self.canonicalize(embeddings, return_details=False)
        latent, nll = self.flow(fused, labels)
        if return_details:
            return latent, nll, {"fused": fused, "base_fused": fused}
        return latent, nll

    @torch.no_grad()
    def score(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.score_differentiable(embeddings).detach()

    def score_differentiable(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = next(iter(embeddings.values())).shape[0]
        labels = torch.zeros(batch, dtype=torch.long, device=next(self.parameters()).device)
        _, nll = self(embeddings, labels)
        return -nll

    def freeze_base(self) -> None:
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(False)
        for parameter in self.flow.parameters():
            parameter.requires_grad_(False)

    def enable_robustness_training(self) -> list[nn.Parameter]:
        self.freeze_base()
        parameters = list(self.robust_fusion.robustness_parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters
