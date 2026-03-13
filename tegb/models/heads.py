from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tegb.types import DirichletOutput


class EvidentialHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> DirichletOutput:
        logits = self.classifier(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        uncertainty = alpha.shape[-1] / (alpha.sum(dim=-1) + 1e-8)
        return DirichletOutput(alpha=alpha, evidence=evidence, uncertainty=uncertainty)


class VisProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

