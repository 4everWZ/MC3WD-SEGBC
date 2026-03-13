from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PersistentTopologyLoss(nn.Module):
    """Topology-aware regularization with PH score + differentiable smoothness proxy."""

    def __init__(self, max_points: int = 128, persistence_threshold: float = 0.05) -> None:
        super().__init__()
        self.max_points = max_points
        self.persistence_threshold = persistence_threshold
        self.last_topo_score = 0.0
        try:
            import gudhi  # type: ignore

            self._gudhi = gudhi
        except Exception:
            self._gudhi = None

    def _ph_score(self, x: np.ndarray) -> float:
        if self._gudhi is None or x.shape[0] < 6:
            return 0.0
        max_edge = float(np.percentile(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1), 80))
        max_edge = max(max_edge, 1e-4)
        rips = self._gudhi.RipsComplex(points=x, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=2)
        pers = st.persistence()
        h1 = [d - b for dim, (b, d) in pers if dim == 1 and np.isfinite(d)]
        if not h1:
            return 0.0
        h1 = np.asarray(h1)
        return float(np.sum(h1[h1 > self.persistence_threshold]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 3:
            self.last_topo_score = 0.0
            return torch.zeros((), device=features.device)

        x = features
        if x.shape[0] > self.max_points:
            idx = torch.randperm(x.shape[0], device=x.device)[: self.max_points]
            x = x[idx]

        # Differentiable proxy: penalize high variance in local kNN distances.
        d = torch.cdist(x, x, p=2)
        knn = torch.topk(d, k=min(6, d.shape[1]), largest=False).values[:, 1:]
        smooth_proxy = torch.var(knn)

        topo_score = self._ph_score(x.detach().cpu().numpy())
        self.last_topo_score = topo_score
        topo_term = torch.tensor(topo_score, device=x.device, dtype=x.dtype)
        return smooth_proxy + 0.1 * topo_term

