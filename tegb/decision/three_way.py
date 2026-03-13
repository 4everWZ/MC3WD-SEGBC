from __future__ import annotations

from typing import Any, List

import torch

from tegb.config.schema import DecisionSection
from tegb.types import (
    DirichletOutput,
    REGION_BOUNDARY,
    REGION_NEGATIVE,
    REGION_POSITIVE,
    ThreeWayOutput,
)


class ThreeWayDecider:
    def __init__(self, cfg: DecisionSection) -> None:
        self.cfg = cfg
        self.last_collision_pairs: List[dict[str, Any]] = []

    def decide(self, out: DirichletOutput, features=None, balls=None) -> ThreeWayOutput:
        if out.alpha.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=out.alpha.device)
            return ThreeWayOutput(empty, empty.bool(), empty.bool())

        probs = out.alpha / torch.clamp(out.alpha.sum(dim=-1, keepdim=True), min=1e-8)
        max_prob = torch.max(probs, dim=-1).values
        uncertainty = out.uncertainty

        labels = torch.full_like(max_prob, fill_value=REGION_BOUNDARY, dtype=torch.long)

        accept = (max_prob >= self.cfg.alpha) & (uncertainty <= self.cfg.uncertainty_accept)
        reject = (max_prob <= self.cfg.beta) | (uncertainty >= self.cfg.uncertainty_reject)

        labels[accept] = REGION_POSITIVE
        labels[reject] = REGION_NEGATIVE

        return ThreeWayOutput(
            region_labels=labels,
            accept_mask=accept,
            reject_mask=reject,
        )
