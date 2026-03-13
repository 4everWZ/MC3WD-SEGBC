from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from tegb.config.schema import GranularSection
from tegb.types import GranularBall


@dataclass
class _Node:
    members: np.ndarray
    depth: int = 0


class SphericalEntropyBuilder:
    """V3 builder: entropy-guided spherical granular balls."""

    def __init__(self, cfg: GranularSection, random_state: int = 42) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self.last_warning_counts: Dict[str, int] = {}
        if cfg.mode == "v3_spherical_entropy" and cfg.split_algorithm != "euclidean_kmeans":
            raise ValueError("v3_spherical_entropy requires granular.split_algorithm='euclidean_kmeans'")

    def _reset_warnings(self) -> None:
        self.last_warning_counts = {
            "split_fallback": 0,
            "empty_child": 0,
            "depth_limit": 0,
            "low_entropy_gain": 0,
        }

    def _purity(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        return float(counts.max() / max(1, counts.sum()))

    def _entropy(self, probs: np.ndarray) -> tuple[np.ndarray, float]:
        avg = np.mean(probs, axis=0)
        avg = np.clip(avg, 1e-12, None)
        avg = avg / np.sum(avg)
        entropy = float(-np.sum(avg * (np.log(avg) / np.log(self.cfg.entropy_log_base))))
        return avg, entropy

    def _ball_from_members(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        hard_labels: np.ndarray,
        members: np.ndarray,
    ) -> GranularBall:
        pts = features[members]
        center = np.mean(pts, axis=0, dtype=np.float64)
        dists = np.linalg.norm(pts - center[None, :], axis=1)
        raw_radius = float(np.max(dists)) if dists.size else 0.0
        if dists.size == 0:
            radius = 0.0
        elif self.cfg.radius_policy == "max":
            radius = raw_radius
        else:
            # Robust radius reduces single-point outlier inflation in high dimensions.
            radius = float(np.percentile(dists, self.cfg.radius_percentile))
            radius = float(np.clip(radius, 0.0, raw_radius))
        p_avg, entropy = self._entropy(probs[members])
        purity = self._purity(hard_labels[members]) if hard_labels.size == features.shape[0] else float(np.max(p_avg))
        return GranularBall(
            center=center.astype(np.float32),
            radius=radius,
            members=members.astype(int).tolist(),
            purity=float(purity),
            topo_state={
                "raw_radius": raw_radius,
                "truncated_radius": radius,
                "mode_v3_sphere": True,
                "radius_policy": self.cfg.radius_policy,
                "radius_percentile": float(self.cfg.radius_percentile),
            },
            semantic_entropy=float(entropy),
            dominant_class=int(np.argmax(p_avg)),
            boundary_score=float(np.exp(-entropy)),
            geometry_type="sphere",
            support_count=int(members.size),
        )

    def _split(self, local: np.ndarray, k: int) -> np.ndarray | None:
        if local.shape[0] < k * max(1, self.cfg.split_min_child):
            self.last_warning_counts["split_fallback"] += 1
            return None
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            return km.fit_predict(local)
        except Exception:
            self.last_warning_counts["split_fallback"] += 1
            return None

    def build(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        hard_labels: np.ndarray | None = None,
    ) -> List[GranularBall]:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array")
        n = features.shape[0]
        if n == 0:
            return []
        if probs.ndim != 2 or probs.shape[0] != n:
            raise ValueError("probs must be a 2D array with same number of rows as features")

        self._reset_warnings()
        if hard_labels is None:
            hard_labels = np.argmax(probs, axis=1).astype(int)
        else:
            hard_labels = hard_labels.reshape(-1).astype(int)

        queue = [_Node(np.arange(n), depth=0)]
        balls: List[GranularBall] = []
        split_k = max(2, int(self.cfg.split_k))

        while queue and len(balls) < self.cfg.max_balls:
            node = queue.pop(0)
            members = node.members

            if members.size < self.cfg.min_members:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            _, entropy = self._entropy(probs[members])
            if entropy <= self.cfg.entropy_threshold:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            if node.depth >= self.cfg.max_split_depth:
                self.last_warning_counts["depth_limit"] += 1
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            local = features[members]
            pred = self._split(local, split_k)
            if pred is None:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            children: List[np.ndarray] = []
            weighted_child_entropy = 0.0
            for c in range(split_k):
                child = members[pred == c]
                if child.size == 0:
                    self.last_warning_counts["empty_child"] += 1
                    continue
                if child.size < self.cfg.split_min_child:
                    self.last_warning_counts["split_fallback"] += 1
                    children = []
                    break
                children.append(child)
                _, child_entropy = self._entropy(probs[child])
                weighted_child_entropy += (child.size / members.size) * child_entropy

            if len(children) <= 1:
                self.last_warning_counts["split_fallback"] += 1
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            entropy_gain = entropy - weighted_child_entropy
            if entropy_gain < self.cfg.min_entropy_gain:
                self.last_warning_counts["low_entropy_gain"] += 1
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            queue.extend([_Node(child, depth=node.depth + 1) for child in children])

        return balls
