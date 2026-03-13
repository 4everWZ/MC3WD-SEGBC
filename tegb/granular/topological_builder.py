from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from tegb.config.schema import GranularSection
from tegb.types import GranularBall


@dataclass
class _Node:
    members: np.ndarray


class TopologicalGranularBuilder:
    """Topological granular-ball construction with PH-aware radius truncation."""

    def __init__(self, cfg: GranularSection, random_state: int = 42) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self._gudhi = None
        try:
            import gudhi  # type: ignore

            self._gudhi = gudhi
        except Exception:
            self._gudhi = None

    def _purity(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        vals, counts = np.unique(labels, return_counts=True)
        return float(counts.max() / max(1, counts.sum()))

    def _phase_transition(self, points: np.ndarray) -> Dict[str, float | bool]:
        if points.shape[0] < 6:
            return {"phase_transition": False, "significant_h1": 0, "h1_max_persistence": 0.0}

        if self._gudhi is not None:
            max_edge = float(np.percentile(pairwise_distances(points), 75))
            max_edge = max(max_edge, 1e-4)
            rips = self._gudhi.RipsComplex(points=points, max_edge_length=max_edge)
            st = rips.create_simplex_tree(max_dimension=2)
            pers = st.persistence()
            h1 = [d - b for dim, (b, d) in pers if dim == 1 and np.isfinite(d)]
            if h1:
                max_h1 = float(np.max(h1))
                significant = int(np.sum(np.array(h1) > self.cfg.ph_persistence_threshold))
                return {
                    "phase_transition": significant > 0,
                    "significant_h1": significant,
                    "h1_max_persistence": max_h1,
                }
            return {"phase_transition": False, "significant_h1": 0, "h1_max_persistence": 0.0}

        # Fallback heuristic when gudhi is unavailable.
        dist = pairwise_distances(points)
        knn = np.sort(dist, axis=1)[:, 1:6]
        ratio = float(np.mean(knn[:, -1] / np.maximum(knn[:, 0], 1e-6)))
        return {
            "phase_transition": ratio > 2.5,
            "significant_h1": int(ratio > 2.5),
            "h1_max_persistence": ratio,
        }

    def _ball_from_members(self, features: np.ndarray, labels: np.ndarray, members: np.ndarray) -> GranularBall:
        pts = features[members]
        center = pts.mean(axis=0)
        distances = np.linalg.norm(pts - center[None, :], axis=1)
        raw_radius = float(np.mean(distances))
        topo = self._phase_transition(pts)
        radius = raw_radius
        if bool(topo["phase_transition"]):
            radius = radius * self.cfg.radius_shrink_factor
        purity = self._purity(labels[members]) if labels.size == features.shape[0] else 0.0
        topo_state: Dict[str, float | int | bool] = {
            "raw_radius": raw_radius,
            "phase_transition": bool(topo["phase_transition"]),
            "significant_h1": int(topo["significant_h1"]),
            "h1_max_persistence": float(topo["h1_max_persistence"]),
        }
        return GranularBall(
            center=center.astype(np.float32),
            radius=float(radius),
            members=members.astype(int).tolist(),
            purity=purity,
            topo_state=topo_state,
            geometry_type="sphere",
            support_count=int(members.size),
        )

    def build(
        self,
        features: np.ndarray,
        probs_or_labels: np.ndarray,
        hard_labels: np.ndarray | None = None,
    ) -> List[GranularBall]:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array")
        n = features.shape[0]
        if n == 0:
            return []

        if hard_labels is not None:
            labels = hard_labels.reshape(-1) if hard_labels.size else np.zeros((n,), dtype=int)
        else:
            labels = probs_or_labels.reshape(-1) if probs_or_labels.size else np.zeros((n,), dtype=int)
        queue = [_Node(members=np.arange(n))]
        balls: List[GranularBall] = []

        while queue and len(balls) < self.cfg.max_balls:
            node = queue.pop(0)
            members = node.members
            if members.size < self.cfg.min_members:
                balls.append(self._ball_from_members(features, labels, members))
                continue
            pur = self._purity(labels[members])
            if pur >= self.cfg.purity_threshold:
                balls.append(self._ball_from_members(features, labels, members))
                continue

            # Impure node: split by local KMeans.
            split_k = max(2, int(self.cfg.split_k))
            km = KMeans(n_clusters=split_k, n_init=5, random_state=self.random_state)
            local = features[members]
            pred = km.fit_predict(local)
            for c in range(split_k):
                child = members[pred == c]
                if child.size == 0:
                    continue
                queue.append(_Node(members=child))

        return balls
