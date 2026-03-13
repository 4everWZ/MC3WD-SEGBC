from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, hard_negative_k: int = 10) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_k = hard_negative_k

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if feats.shape[0] < 3 or labels.numel() == 0:
            return torch.zeros((), device=feats.device)
        z = F.normalize(feats, dim=-1)
        sim = torch.matmul(z, z.t()) / self.temperature

        losses = []
        for i in range(sim.shape[0]):
            same = torch.where(labels == labels[i])[0]
            diff = torch.where(labels != labels[i])[0]
            same = same[same != i]
            if same.numel() == 0 or diff.numel() == 0:
                continue

            pos_idx = same[torch.randint(0, same.numel(), (1,), device=feats.device)].item()
            pos_logit = sim[i, pos_idx]

            neg_sims = sim[i, diff]
            k = min(self.hard_negative_k, neg_sims.numel())
            hard_vals, _ = torch.topk(neg_sims, k=k, largest=True)

            num = torch.exp(pos_logit)
            den = num + torch.sum(torch.exp(hard_vals))
            losses.append(-torch.log(torch.clamp(num / torch.clamp(den, min=1e-8), min=1e-8)))

        if not losses:
            return torch.zeros((), device=feats.device)
        return torch.mean(torch.stack(losses))

