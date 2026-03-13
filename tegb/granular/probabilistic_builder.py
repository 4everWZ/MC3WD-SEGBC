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


class ProbabilisticGranularBuilder:
    """Entropy-guided probabilistic ellipsoid granular builder.

    Research-grade details:
    - covariance-aware Mahalanobis geometry
    - semantic entropy split stopping criterion
    - chi-square confidence boundary
    - optional PH-aware boundary truncation
    - iterative Mahalanobis KMeans refinement
    """

    def __init__(self, cfg: GranularSection, random_state: int = 42) -> None:
        self.cfg = cfg
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.last_warning_counts: Dict[str, int] = {}
        self._chi2 = None
        self._gudhi = None
        self._ledoit_wolf = None
        self._oas = None
        try:
            from scipy.stats import chi2  # type: ignore

            self._chi2 = chi2
        except Exception:
            self._chi2 = None
        try:
            import gudhi  # type: ignore

            self._gudhi = gudhi
        except Exception:
            self._gudhi = None
        try:
            from sklearn.covariance import LedoitWolf, OAS  # type: ignore

            self._ledoit_wolf = LedoitWolf
            self._oas = OAS
        except Exception:
            self._ledoit_wolf = None
            self._oas = None

    def _reset_warnings(self) -> None:
        self.last_warning_counts = {
            "cov_fallback": 0,
            "chi2_fallback": 0,
            "split_fallback": 0,
            "split_non_converged": 0,
            "ph_fallback": 0,
        }

    def _purity(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        return float(counts.max() / max(1, counts.sum()))

    def _estimate_covariance(self, points: np.ndarray) -> np.ndarray:
        d = points.shape[1]
        if points.shape[0] <= 1:
            return np.eye(d, dtype=np.float64) * max(self.cfg.cov_jitter, self.cfg.min_eig)

        cov: np.ndarray
        if self.cfg.covariance_estimator == "ledoit_wolf" and self._ledoit_wolf is not None:
            cov = self._ledoit_wolf().fit(points).covariance_
        elif self.cfg.covariance_estimator == "oas" and self._oas is not None:
            cov = self._oas().fit(points).covariance_
        else:
            if self.cfg.covariance_estimator in {"ledoit_wolf", "oas"}:
                self.last_warning_counts["cov_fallback"] += 1
            cov = np.cov(points, rowvar=False)
            if np.asarray(cov).ndim == 0:
                cov = np.eye(d, dtype=np.float64) * float(cov)
        return np.asarray(cov, dtype=np.float64)

    def _safe_covariance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        d = points.shape[1]
        cov = self._estimate_covariance(points)
        if cov.shape != (d, d):
            self.last_warning_counts["cov_fallback"] += 1
            cov = np.eye(d, dtype=np.float64)

        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.cfg.cov_shrinkage) * cov + self.cfg.cov_shrinkage * diag
        cov = cov + np.eye(d, dtype=np.float64) * self.cfg.cov_jitter

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, self.cfg.min_eig)
            cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            inv_cov = np.linalg.pinv(cov_psd, rcond=self.cfg.min_eig)
            d_eff = int(max(1, np.sum(eigvals > self.cfg.min_eig * 10.0)))
            return cov_psd, inv_cov, eigvals, d_eff
        except np.linalg.LinAlgError:
            self.last_warning_counts["cov_fallback"] += 1
            safe = np.diag(np.maximum(np.diag(cov), self.cfg.min_eig))
            safe = safe + np.eye(d, dtype=np.float64) * self.cfg.cov_jitter
            inv_safe = np.linalg.pinv(safe, rcond=self.cfg.min_eig)
            eigvals = np.diag(safe)
            d_eff = int(max(1, np.sum(eigvals > self.cfg.min_eig * 10.0)))
            return safe, inv_safe, eigvals, d_eff

    def _mahalanobis_sq(self, points: np.ndarray, center: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
        delta = points - center[None, :]
        return np.einsum("nd,dd,nd->n", delta, inv_cov, delta)

    def _entropy(self, probs: np.ndarray) -> tuple[np.ndarray, float]:
        avg = np.mean(probs, axis=0)
        avg = np.clip(avg, 1e-12, None)
        avg = avg / np.sum(avg)
        entropy = float(-np.sum(avg * np.log(avg)))
        return avg, entropy

    def _chi2_threshold(self, d_eff: int, distances_sq: np.ndarray) -> float:
        if self._chi2 is not None:
            return float(self._chi2.ppf(self.cfg.confidence_level, df=max(1, d_eff)))
        self.last_warning_counts["chi2_fallback"] += 1
        return float(np.quantile(distances_sq, self.cfg.confidence_level))

    def _phase_transition(self, points: np.ndarray) -> Dict[str, float | int | bool]:
        if points.shape[0] < 6:
            return {"phase_transition": False, "significant_h1": 0, "h1_max_persistence": 0.0}

        local = points
        if local.shape[0] > self.cfg.ph_max_points:
            idx = self._rng.choice(local.shape[0], size=self.cfg.ph_max_points, replace=False)
            local = local[idx]

        if self._gudhi is not None:
            dist = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
            max_edge = float(np.percentile(dist, 75))
            max_edge = max(max_edge, 1e-4)
            rips = self._gudhi.RipsComplex(points=local, max_edge_length=max_edge)
            st = rips.create_simplex_tree(max_dimension=2)
            pers = st.persistence()
            h1 = [d - b for dim, (b, d) in pers if dim == 1 and np.isfinite(d)]
            if h1:
                h1_arr = np.asarray(h1, dtype=np.float64)
                significant = int(np.sum(h1_arr > self.cfg.ph_persistence_threshold))
                return {
                    "phase_transition": significant > 0,
                    "significant_h1": significant,
                    "h1_max_persistence": float(np.max(h1_arr)),
                }
            return {"phase_transition": False, "significant_h1": 0, "h1_max_persistence": 0.0}

        self.last_warning_counts["ph_fallback"] += 1
        dist = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
        knn = np.sort(dist, axis=1)[:, 1 : min(6, dist.shape[1])]
        if knn.shape[1] < 2:
            return {"phase_transition": False, "significant_h1": 0, "h1_max_persistence": 0.0}
        ratio = float(np.mean(knn[:, -1] / np.maximum(knn[:, 0], 1e-8)))
        return {
            "phase_transition": ratio > 2.5,
            "significant_h1": int(ratio > 2.5),
            "h1_max_persistence": ratio,
        }

    def _split_mahalanobis(self, local: np.ndarray, k: int) -> np.ndarray | None:
        k = max(2, int(k))
        if local.shape[0] < k * self.cfg.split_min_child:
            self.last_warning_counts["split_fallback"] += 1
            return None

        # Initialize by Euclidean KMeans, then refine with local Mahalanobis assignments.
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(local)
        prev_obj = np.inf
        converged = False
        for _ in range(self.cfg.split_max_iter):
            centers = []
            inv_covs = []
            valid = True
            for c in range(k):
                pts = local[labels == c]
                if pts.shape[0] < self.cfg.split_min_child:
                    valid = False
                    break
                center = np.mean(pts, axis=0)
                _, inv_cov, _, _ = self._safe_covariance(pts)
                centers.append(center)
                inv_covs.append(inv_cov)
            if not valid:
                self.last_warning_counts["split_fallback"] += 1
                return None

            d2_cols = []
            for c in range(k):
                d2_cols.append(self._mahalanobis_sq(local, centers[c], inv_covs[c]))
            d2 = np.stack(d2_cols, axis=1)
            new_labels = np.argmin(d2, axis=1)
            counts = np.bincount(new_labels, minlength=k)
            if np.any(counts < self.cfg.split_min_child):
                self.last_warning_counts["split_fallback"] += 1
                return None

            obj = float(np.mean(np.min(d2, axis=1)))
            moved = float(np.mean(new_labels != labels))
            labels = new_labels
            if abs(prev_obj - obj) <= self.cfg.split_tolerance and moved == 0.0:
                converged = True
                break
            prev_obj = obj

        if not converged:
            self.last_warning_counts["split_non_converged"] += 1
        return labels

    def _ball_from_members(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        hard_labels: np.ndarray,
        members: np.ndarray,
    ) -> GranularBall:
        pts = features[members]
        local_probs = probs[members]
        center = pts.mean(axis=0).astype(np.float64)
        cov, inv_cov, eigvals, d_eff = self._safe_covariance(pts)
        p_avg, entropy = self._entropy(local_probs)
        d2 = self._mahalanobis_sq(pts, center, inv_cov)
        chi2_threshold_raw = self._chi2_threshold(d_eff, d2)
        chi2_threshold = chi2_threshold_raw
        topo = self._phase_transition(pts)
        if bool(topo["phase_transition"]):
            chi2_threshold = chi2_threshold * (self.cfg.radius_shrink_factor**2)
        raw_radius = float(np.sqrt(max(chi2_threshold_raw, 0.0)))
        radius = float(np.sqrt(max(chi2_threshold, 0.0)))
        purity = self._purity(hard_labels[members]) if hard_labels.size == features.shape[0] else float(np.max(p_avg))

        topo_state: Dict[str, float | int | bool] = {
            "raw_radius": raw_radius,
            "truncated_radius": radius,
            "raw_chi2_threshold": float(chi2_threshold_raw),
            "truncated_chi2_threshold": float(chi2_threshold),
            "phase_transition": bool(topo["phase_transition"]),
            "significant_h1": int(topo["significant_h1"]),
            "h1_max_persistence": float(topo["h1_max_persistence"]),
            "mode_prob_ellipsoid": True,
            "d_eff": d_eff,
            "eig_min": float(np.min(eigvals)),
            "eig_max": float(np.max(eigvals)),
        }
        return GranularBall(
            center=center.astype(np.float32),
            radius=radius,
            members=members.astype(int).tolist(),
            purity=float(purity),
            topo_state=topo_state,
            covariance=cov.astype(np.float32),
            inv_covariance=inv_cov.astype(np.float32),
            chi2_threshold=float(chi2_threshold),
            semantic_entropy=float(entropy),
            dominant_class=int(np.argmax(p_avg)),
            confidence_level=float(self.cfg.confidence_level),
            boundary_score=float(np.exp(-entropy)),
            geometry_type="ellipsoid",
            support_count=int(members.size),
        )

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

        queue = [_Node(members=np.arange(n), depth=0)]
        balls: List[GranularBall] = []
        split_k = max(2, int(self.cfg.split_k))
        min_split_size = split_k * max(self.cfg.min_members, self.cfg.split_min_child)

        while queue and len(balls) < self.cfg.max_balls:
            node = queue.pop(0)
            members = node.members
            if members.size < self.cfg.min_members:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            pts = features[members]
            _, entropy = self._entropy(probs[members])
            should_split = entropy > self.cfg.entropy_threshold and members.size >= min_split_size

            if not should_split:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue

            pred = self._split_mahalanobis(pts, split_k)
            if pred is None:
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue
            child_nodes = []
            for c in range(split_k):
                child = members[pred == c]
                if child.size == 0:
                    continue
                child_nodes.append(_Node(members=child, depth=node.depth + 1))
            if len(child_nodes) <= 1:
                self.last_warning_counts["split_fallback"] += 1
                balls.append(self._ball_from_members(features, probs, hard_labels, members))
                continue
            queue.extend(child_nodes)

        return balls
