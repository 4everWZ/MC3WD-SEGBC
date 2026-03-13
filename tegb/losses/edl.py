from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tegb.types import DirichletOutput


class EvidentialLoss(nn.Module):
    """EDL classification objective with KL regularization to uniform prior."""

    def __init__(self, kl_weight: float = 1e-3) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def _kl_dirichlet_uniform(self, alpha: torch.Tensor) -> torch.Tensor:
        k = alpha.shape[-1]
        beta = torch.ones((1, k), device=alpha.device, dtype=alpha.dtype)
        sum_alpha = alpha.sum(dim=1, keepdim=True)
        sum_beta = beta.sum(dim=1, keepdim=True)

        ln_b_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        ln_b_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(sum_beta)
        digamma_sum_alpha = torch.digamma(sum_alpha)
        digamma_alpha = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim=1, keepdim=True)
        kl = kl + ln_b_alpha + ln_b_beta
        return kl.mean()

    def forward(self, out: DirichletOutput, targets: torch.Tensor) -> torch.Tensor:
        if targets.numel() == 0 or out.alpha.numel() == 0:
            return torch.zeros((), device=out.alpha.device)

        alpha = out.alpha
        n_classes = alpha.shape[-1]
        target_onehot = F.one_hot(targets, num_classes=n_classes).float()
        s = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / torch.clamp(s, min=1e-8)

        var = alpha * (s - alpha) / (torch.clamp(s * s * (s + 1.0), min=1e-8))
        mse_term = torch.sum((target_onehot - probs) ** 2, dim=1, keepdim=True)
        var_term = torch.sum(var, dim=1, keepdim=True)

        ll = (mse_term + var_term).mean()
        alpha_tilde = (alpha - 1.0) * (1.0 - target_onehot) + 1.0
        kl = self._kl_dirichlet_uniform(alpha_tilde)
        return ll + self.kl_weight * kl

