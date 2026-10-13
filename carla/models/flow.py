"""Vector normalizing flow used by CARLA (compatible with the v1 density flow)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

_LOG2 = math.log(2.0)
_LOG2PI = math.log(2.0 * math.pi)


def gaussian_logp(mean: torch.Tensor, logs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    value = -0.5 * (2.0 * logs + (x - mean).square() / torch.exp(2.0 * logs) + _LOG2PI)
    return value.sum(dim=1)


class ActNorm1d(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("inited", torch.tensor(0, dtype=torch.uint8))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.logs = nn.Parameter(torch.zeros(1, dim))
        self.scale = scale

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        bias = -x.mean(0, keepdim=True)
        var = (x + bias).square().mean(0, keepdim=True)
        self.bias.copy_(bias)
        self.logs.copy_(torch.log(self.scale / (var.sqrt() + 1e-6)))
        self.inited.fill_(1)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False):
        if self.inited.item() == 0:
            self._initialize(x)
        if reverse:
            return x * torch.exp(-self.logs) - self.bias, logdet - self.logs.sum()
        return (x + self.bias) * torch.exp(self.logs), logdet + self.logs.sum()


class Permute1d(nn.Module):
    def __init__(self, dim: int, mode: str = "reverse") -> None:
        super().__init__()
        perm = torch.randperm(dim) if mode == "random" else torch.arange(dim - 1, -1, -1)
        inv = torch.zeros_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", inv)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return x[:, self.inv] if reverse else x[:, self.perm]


class AffineCoupling1d(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.split = dim // 2
        out_dim = dim - self.split
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False):
        z1, z2 = x[:, :self.split], x[:, self.split:]
        shift, log_scale = self.net(z1).chunk(2, dim=1)
        scale = torch.sigmoid(log_scale + 2.0) + 1e-6
        if reverse:
            z2 = z2 / scale - shift
            logdet = logdet - scale.log().sum(1)
        else:
            z2 = (z2 + shift) * scale
            logdet = logdet + scale.log().sum(1)
        return torch.cat((z1, z2), 1), logdet


class FlowStep1d(nn.Module):
    def __init__(self, dim: int, hidden: int, permutation: str) -> None:
        super().__init__()
        self.actnorm = ActNorm1d(dim)
        self.permute = Permute1d(dim, permutation)
        self.coupling = AffineCoupling1d(dim, hidden)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False):
        if reverse:
            x, logdet = self.coupling(x, logdet, True)
            x = self.permute(x, True)
            return self.actnorm(x, logdet, True)
        x, logdet = self.actnorm(x, logdet)
        x = self.permute(x)
        return self.coupling(x, logdet)


class VectorGlow(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        K: int = 8,
        hidden: int = 512,
        R: float = 5.0,
        permutation: str = "reverse",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.R = float(R)
        self.flows = nn.ModuleList(FlowStep1d(dim, hidden, permutation) for _ in range(K))

    def forward_logdet(self, x: torch.Tensor):
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for flow in self.flows:
            z, logdet = flow(z, logdet)
        return z, logdet

    def prior_mean(
        self, labels: Optional[torch.Tensor], batch: int, device: torch.device
    ) -> torch.Tensor:
        if labels is None:
            return torch.zeros(batch, self.dim, device=device)
        sign = torch.where(labels == 0, 1.0, -1.0).to(device)
        return (sign * self.R).unsqueeze(1).expand(batch, self.dim)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        z, logdet = self.forward_logdet(x)
        mean = self.prior_mean(labels, x.shape[0], x.device)
        logp = gaussian_logp(mean, torch.zeros_like(mean), z)
        return z, -(logdet + logp) / (_LOG2 * self.dim)
