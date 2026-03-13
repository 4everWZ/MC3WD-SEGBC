from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances

from tegb.types import GranularBall


def _as_float(x: float | np.floating[Any] | np.ndarray) -> float:
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return 0.0
        return float(x.reshape(-1)[0])
    return float(x)


def _stable_eigvals(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] < 2:
        return np.asarray([1e-12], dtype=np.float64)
    cov = np.cov(points, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-8 * np.eye(cov.shape[0], dtype=np.float64)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return np.asarray([1e-12], dtype=np.float64)
    eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 1e-12, None)
    return eigvals


def _anisotropy_ratio(eigvals: np.ndarray) -> float:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvals.size == 0:
        return 1.0
    return float(np.max(eigvals) / np.maximum(np.min(eigvals), 1e-12))


def _sphericity(eigvals: np.ndarray) -> float:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvals.size == 0:
        return 0.0
    gm = np.exp(np.mean(np.log(np.clip(eigvals, 1e-12, None))))
    am = np.mean(eigvals)
    if am <= 1e-12:
        return 0.0
    return float(np.clip(gm / am, 0.0, 1.0))


def build_point_to_ball_index(num_points: int, balls: List[GranularBall]) -> tuple[np.ndarray, int]:
    assignment = np.full((int(num_points),), -1, dtype=np.int64)
    overlap_conflicts = 0
    for b_idx, b in enumerate(balls):
        members = np.asarray(b.members, dtype=np.int64)
        if members.size == 0:
            continue
        valid = members[(members >= 0) & (members < assignment.shape[0])]
        if valid.size == 0:
            continue
        overlap_conflicts += int(np.sum(assignment[valid] >= 0))
        assignment[valid] = int(b_idx)
    return assignment, int(overlap_conflicts)


def _subsample_indices(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=cap, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    n = int(points.shape[0])
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k_eff = int(max(1, min(k, n - 1)))
    dist = pairwise_distances(points, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    inds = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.arange(n, dtype=np.int64)[:, None]
    local_order = np.argsort(dist[row, inds], axis=1)
    ordered = inds[row, local_order]
    return ordered.astype(np.int64, copy=False)


def compute_manifold_diagnostics(
    features: np.ndarray,
    balls: List[GranularBall],
    labels: np.ndarray | None,
    sample_cap: int = 2000,
    knn_k: int = 15,
    min_ball_members: int = 8,
    seed: int = 42,
    topk_ball_details: int = 80,
) -> Dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    n = int(features.shape[0])
    assignment, overlap_conflicts = build_point_to_ball_index(n, balls)

    sampled_idx = _subsample_indices(n, int(sample_cap), seed=seed)
    sampled_features = features[sampled_idx] if sampled_idx.size else np.zeros((0, features.shape[1]), dtype=np.float64)
    sampled_assignment = assignment[sampled_idx] if sampled_idx.size else np.zeros((0,), dtype=np.int64)

    sampled_labels = None
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] == n:
            sampled_labels = labels[sampled_idx]

    global_eigs = _stable_eigvals(sampled_features)
    ball_rows: List[Dict[str, float | int | str | None]] = []
    ball_ratios: List[float] = []
    ball_sphericities: List[float] = []

    min_members = max(2, int(min_ball_members))
    for idx, b in enumerate(balls):
        members = np.asarray(b.members, dtype=np.int64)
        valid = members[(members >= 0) & (members < n)]
        if valid.size < min_members:
            continue
        eigs = _stable_eigvals(features[valid])
        ratio = _anisotropy_ratio(eigs)
        sph = _sphericity(eigs)
        ball_ratios.append(ratio)
        ball_sphericities.append(sph)
        ball_rows.append(
            {
                "ball_index": int(idx),
                "members": int(valid.size),
                "dominant_class": (None if b.dominant_class is None else int(b.dominant_class)),
                "geometry_type": (b.geometry_type or "unknown"),
                "anisotropy_ratio": float(ratio),
                "sphericity": float(sph),
                "radius": float(max(float(b.radius), 0.0)),
            }
        )

    ball_rows_sorted = sorted(ball_rows, key=lambda x: float(x["anisotropy_ratio"]), reverse=True)
    if topk_ball_details > 0:
        ball_rows_sorted = ball_rows_sorted[: int(topk_ball_details)]

    knn_neighbors = _knn_indices(sampled_features, k=int(knn_k))
    if knn_neighbors.size == 0:
        knn_cross_ball_ratio = 0.0
        knn_same_label_cross_ball_ratio = 0.0
        knn_diff_label_same_ball_ratio = 0.0
        knn_unassigned_ratio = 0.0
        knn_edges = 0
    else:
        rows = np.repeat(np.arange(knn_neighbors.shape[0], dtype=np.int64), knn_neighbors.shape[1])
        cols = knn_neighbors.reshape(-1)
        a_r = sampled_assignment[rows]
        a_c = sampled_assignment[cols]
        cross_ball = a_r != a_c
        unassigned = (a_r < 0) | (a_c < 0)
        knn_edges = int(rows.size)
        knn_cross_ball_ratio = _as_float(np.mean(cross_ball)) if knn_edges > 0 else 0.0
        knn_unassigned_ratio = _as_float(np.mean(unassigned)) if knn_edges > 0 else 0.0
        if sampled_labels is not None and sampled_labels.shape[0] == sampled_features.shape[0]:
            same_label = sampled_labels[rows] == sampled_labels[cols]
            diff_label = ~same_label
            if np.any(same_label):
                knn_same_label_cross_ball_ratio = _as_float(np.mean(cross_ball[same_label]))
            else:
                knn_same_label_cross_ball_ratio = 0.0
            if np.any(diff_label):
                knn_diff_label_same_ball_ratio = _as_float(np.mean((~cross_ball)[diff_label]))
            else:
                knn_diff_label_same_ball_ratio = 0.0
        else:
            knn_same_label_cross_ball_ratio = 0.0
            knn_diff_label_same_ball_ratio = 0.0

    summary = {
        "n_points": int(n),
        "sampled_points": int(sampled_features.shape[0]),
        "knn_k": int(max(1, min(int(knn_k), max(sampled_features.shape[0] - 1, 1)))),
        "knn_edges": int(knn_edges),
        "ball_count_total": int(len(balls)),
        "ball_count_analyzed": int(len(ball_ratios)),
        "assignment_uncovered_count": int(np.sum(assignment < 0)),
        "assignment_overlap_conflicts": int(overlap_conflicts),
        "anisotropy_global_ratio": float(_anisotropy_ratio(global_eigs)),
        "sphericity_global": float(_sphericity(global_eigs)),
        "anisotropy_ball_mean": float(np.mean(ball_ratios)) if ball_ratios else 1.0,
        "anisotropy_ball_p90": float(np.percentile(ball_ratios, 90.0)) if ball_ratios else 1.0,
        "sphericity_ball_mean": float(np.mean(ball_sphericities)) if ball_sphericities else 0.0,
        "anisotropy_ball_gt10_rate": float(np.mean(np.asarray(ball_ratios) > 10.0)) if ball_ratios else 0.0,
        "knn_cross_ball_ratio": float(knn_cross_ball_ratio),
        "knn_same_label_cross_ball_ratio": float(knn_same_label_cross_ball_ratio),
        "knn_diff_label_same_ball_ratio": float(knn_diff_label_same_ball_ratio),
        "knn_unassigned_ratio": float(knn_unassigned_ratio),
    }
    return {
        "summary": summary,
        "worst_balls_by_anisotropy": ball_rows_sorted,
    }
