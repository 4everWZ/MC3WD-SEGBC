from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GromovWassersteinLoss(nn.Module):
    """Differentiable GW proxy with optional POT metric logging."""

    def __init__(self, entropy_reg: float = 5e-2, metric_weight: float = 0.1) -> None:
        super().__init__()
        self.entropy_reg = entropy_reg
        self.metric_weight = metric_weight
        self.last_metric = 0.0
        try:
            import ot  # type: ignore

            self._ot = ot
        except Exception:
            self._ot = None

    def _pairwise(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(x, x, p=2)
        return d / (torch.mean(d).detach() + 1e-8)

    def _pot_metric(self, c1: np.ndarray, c2: np.ndarray) -> float:
        if self._ot is None:
            return 0.0
        n = c1.shape[0]
        p = np.ones((n,)) / n
        q = np.ones((n,)) / n
        try:
            val = self._ot.gromov.gromov_wasserstein2(
                c1,
                c2,
                p,
                q,
                loss_fun="square_loss",
                epsilon=self.entropy_reg,
            )
            return float(val)
        except TypeError:
            # Older POT versions may not support epsilon in this call.
            val = self._ot.gromov.gromov_wasserstein2(c1, c2, p, q, loss_fun="square_loss")
            return float(val)

    def forward(self, high_features: torch.Tensor, low_features: torch.Tensor) -> torch.Tensor:
        n = min(high_features.shape[0], low_features.shape[0])
        if n < 2:
            self.last_metric = 0.0
            return torch.zeros((), device=high_features.device)
        xh = high_features[:n]
        xl = low_features[:n]
        c1 = self._pairwise(xh)
        c2 = self._pairwise(xl)
        proxy = torch.mean((c1 - c2) ** 2)

        metric = self._pot_metric(
            c1.detach().cpu().numpy().astype(np.float64),
            c2.detach().cpu().numpy().astype(np.float64),
        )
        self.last_metric = metric
        metric_term = torch.tensor(metric, device=proxy.device, dtype=proxy.dtype)
        return proxy + self.metric_weight * metric_term

