from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tegb.types import DirichletOutput


class DetectionSurrogateLoss(nn.Module):
    """Surrogate detection classification loss over object crops."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.class_weights = class_weights

    def forward(self, out: DirichletOutput, targets: torch.Tensor) -> torch.Tensor:
        if targets.numel() == 0 or out.alpha.numel() == 0:
            return torch.zeros((), device=out.alpha.device)
        probs = out.alpha / out.alpha.sum(dim=-1, keepdim=True)
        log_probs = torch.log(torch.clamp(probs, min=1e-8))
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=out.alpha.device, dtype=out.alpha.dtype)
        return F.nll_loss(log_probs, targets, reduction="mean", weight=weight)
