from __future__ import annotations

from itertools import combinations
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

from tegb.config.schema import DecisionSection
from tegb.types import (
    DirichletOutput,
    GranularBall,
    REGION_BOUNDARY,
    REGION_NEGATIVE,
    REGION_POSITIVE,
    ThreeWayOutput,
)


class SphericalCollisionThreeWayDecider:
    """V3 pure-geometric three-way decision for spherical granular balls."""

    def __init__(self, cfg: DecisionSection) -> None:
        self.cfg = cfg
        self.last_collision_pairs: List[dict[str, Any]] = []
        self.last_support_count: np.ndarray = np.zeros((0,), dtype=np.int64)

    def _empty_output(self, out: DirichletOutput) -> ThreeWayOutput:
        device = out.alpha.device
        n = int(out.alpha.shape[0])
        labels = torch.full((n,), fill_value=REGION_NEGATIVE, dtype=torch.long, device=device)
        accept = labels == REGION_POSITIVE
        reject = labels == REGION_NEGATIVE
        collision = torch.zeros((n,), dtype=torch.bool, device=device)
        nearest = torch.full((n,), fill_value=-1, dtype=torch.long, device=device)
        euclidean = torch.full((n,), fill_value=float("inf"), dtype=torch.float32, device=device)
        self.last_collision_pairs = []
        self.last_support_count = np.zeros((n,), dtype=np.int64)
        return ThreeWayOutput(
            region_labels=labels,
            accept_mask=accept,
            reject_mask=reject,
            collision_mask=collision,
            nearest_ball_index=nearest,
            mahalanobis_score=euclidean,
        )

    def _centers_radii_classes(self, balls: Sequence[GranularBall]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = np.stack([np.asarray(b.center, dtype=np.float32) for b in balls], axis=0)
        radii = np.asarray([max(float(b.radius), 0.0) for b in balls], dtype=np.float32)
        cls = np.asarray(
            [(-1 if b.dominant_class is None else int(b.dominant_class)) for b in balls],
            dtype=np.int64,
        )
        return centers, radii, cls

    def _chunk_size(self, num_balls: int, target_cells: int = 2_000_000) -> int:
        if num_balls <= 0:
            return 1024
        return max(64, int(target_cells // max(1, int(num_balls))))

    def _collision_pairs(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        classes: np.ndarray,
    ) -> set[Tuple[int, int]]:
        pairs: List[dict[str, Any]] = []
        pair_set: set[Tuple[int, int]] = set()
        for i in range(centers.shape[0]):
            for j in range(i + 1, centers.shape[0]):
                if classes[i] < 0 or classes[j] < 0 or classes[i] == classes[j]:
                    continue
                center_distance = float(np.linalg.norm(centers[i] - centers[j]))
                threshold = float(self.cfg.collision_gamma * (radii[i] + radii[j]))
                if center_distance < threshold:
                    pair = (i, j)
                    pair_set.add(pair)
                    pairs.append(
                        {
                            "i": i,
                            "j": j,
                            "class_i": int(classes[i]),
                            "class_j": int(classes[j]),
                            "center_distance": center_distance,
                            "threshold": threshold,
                            "gamma": float(self.cfg.collision_gamma),
                            "metric": "spherical_l2_collision",
                        }
                    )
        self.last_collision_pairs = pairs
        return pair_set

    def decide(self, out: DirichletOutput, features=None, balls=None) -> ThreeWayOutput:
        if out.alpha.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=out.alpha.device)
            return ThreeWayOutput(
                region_labels=empty,
                accept_mask=empty.bool(),
                reject_mask=empty.bool(),
                collision_mask=empty.bool(),
                nearest_ball_index=empty.clone(),
                mahalanobis_score=empty.float(),
            )

        if features is None or balls is None or len(balls) == 0:
            return self._empty_output(out)

        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("features must be a 2D array for v3_spherical_collision")

        centers, radii, classes = self._centers_radii_classes(balls)
        if centers.shape[1] != x.shape[1]:
            raise ValueError(
                f"feature dim mismatch between samples ({x.shape[1]}) and balls ({centers.shape[1]})"
            )

        collision_pair_set = self._collision_pairs(centers, radii, classes)
        labels_np = np.full((x.shape[0],), fill_value=REGION_NEGATIVE, dtype=np.int64)
        collision_mask_np = np.zeros((x.shape[0],), dtype=bool)
        support_count = np.zeros((x.shape[0],), dtype=np.int64)
        nearest_idx_np = np.full((x.shape[0],), fill_value=-1, dtype=np.int64)
        nearest_dist_np = np.full((x.shape[0],), fill_value=np.inf, dtype=np.float32)

        chunk = self._chunk_size(int(centers.shape[0]))
        for start in range(0, x.shape[0], chunk):
            end = min(x.shape[0], start + chunk)
            x_chunk = x[start:end]
            delta = x_chunk[:, None, :] - centers[None, :, :]
            dist = np.linalg.norm(delta, axis=2)
            inside = dist <= radii[None, :]

            support_chunk = np.sum(inside, axis=1).astype(np.int64)
            support_count[start:end] = support_chunk
            nearest_chunk = np.argmin(dist, axis=1).astype(np.int64)
            nearest_idx_np[start:end] = nearest_chunk
            nearest_dist_np[start:end] = dist[np.arange(dist.shape[0]), nearest_chunk].astype(np.float32)

            for local_i in range(x_chunk.shape[0]):
                global_i = start + local_i
                cnt = int(support_chunk[local_i])
                if cnt == 0:
                    labels_np[global_i] = REGION_NEGATIVE
                    continue
                if cnt == 1:
                    labels_np[global_i] = REGION_POSITIVE
                    continue

                covered = np.where(inside[local_i])[0]
                covered_classes = classes[covered]
                unique_classes = np.unique(covered_classes[covered_classes >= 0])
                if unique_classes.size <= 1:
                    labels_np[global_i] = REGION_POSITIVE
                    continue

                is_collision_boundary = False
                for a, b in combinations(covered.tolist(), 2):
                    pair = (a, b) if a < b else (b, a)
                    if pair in collision_pair_set:
                        is_collision_boundary = True
                        break

                if is_collision_boundary:
                    labels_np[global_i] = REGION_BOUNDARY
                    collision_mask_np[global_i] = True
                else:
                    labels_np[global_i] = REGION_POSITIVE

        self.last_support_count = support_count

        device = out.alpha.device
        labels = torch.from_numpy(labels_np).to(device)
        collision_mask = torch.from_numpy(collision_mask_np).to(device)
        nearest_idx = torch.from_numpy(nearest_idx_np).to(device)
        nearest_dist = torch.from_numpy(nearest_dist_np).to(device)

        return ThreeWayOutput(
            region_labels=labels,
            accept_mask=(labels == REGION_POSITIVE),
            reject_mask=(labels == REGION_NEGATIVE),
            collision_mask=collision_mask,
            nearest_ball_index=nearest_idx,
            mahalanobis_score=nearest_dist,
        )
