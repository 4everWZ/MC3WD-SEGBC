from __future__ import annotations

from typing import Any, List

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


class AdaptiveThreeWayDecider:
    """Collision-aware three-way decision with legacy uncertainty thresholds."""

    def __init__(self, cfg: DecisionSection) -> None:
        self.cfg = cfg
        self.last_collision_pairs: List[dict[str, Any]] = []

    def _legacy(self, out: DirichletOutput) -> ThreeWayOutput:
        probs = out.alpha / torch.clamp(out.alpha.sum(dim=-1, keepdim=True), min=1e-8)
        max_prob = torch.max(probs, dim=-1).values
        uncertainty = out.uncertainty

        labels = torch.full_like(max_prob, fill_value=REGION_BOUNDARY, dtype=torch.long)
        accept = (max_prob >= self.cfg.alpha) & (uncertainty <= self.cfg.uncertainty_accept)
        reject = (max_prob <= self.cfg.beta) | (uncertainty >= self.cfg.uncertainty_reject)

        labels[accept] = REGION_POSITIVE
        labels[reject] = REGION_NEGATIVE
        collision = torch.zeros_like(accept, dtype=torch.bool)
        nearest = torch.full_like(labels, fill_value=-1, dtype=torch.long)
        maha = torch.full_like(max_prob, fill_value=float("inf"), dtype=torch.float32)
        return ThreeWayOutput(
            region_labels=labels,
            accept_mask=(labels == REGION_POSITIVE),
            reject_mask=(labels == REGION_NEGATIVE),
            collision_mask=collision,
            nearest_ball_index=nearest,
            mahalanobis_score=maha,
        )

    def _ball_threshold(self, b: GranularBall) -> float:
        if b.chi2_threshold is not None:
            return float(b.chi2_threshold)
        return float(max(b.radius * b.radius, 1e-6))

    def _stable_logdet(self, s: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0:
            return float("inf")
        return float(logdet)

    def _bhattacharyya_distance(self, mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
        d = mu1.shape[0]
        eye = np.eye(d, dtype=np.float64) * 1e-6
        s = 0.5 * (s1 + s2) + eye
        inv_s = np.linalg.pinv(s, rcond=1e-8)
        diff = (mu1 - mu2).reshape(-1, 1)
        term1 = 0.125 * float((diff.T @ inv_s @ diff).item())
        logdet_s = self._stable_logdet(s)
        logdet_s1 = self._stable_logdet(s1 + eye)
        logdet_s2 = self._stable_logdet(s2 + eye)
        if not np.isfinite(logdet_s) or not np.isfinite(logdet_s1) or not np.isfinite(logdet_s2):
            return float("inf")
        term2 = 0.5 * (logdet_s - 0.5 * (logdet_s1 + logdet_s2))
        return float(max(term1 + term2, 0.0))

    def _sym_kl_distance(self, mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
        d = mu1.shape[0]
        eye = np.eye(d, dtype=np.float64) * 1e-6
        s1e = s1 + eye
        s2e = s2 + eye
        inv_s1 = np.linalg.pinv(s1e, rcond=1e-8)
        inv_s2 = np.linalg.pinv(s2e, rcond=1e-8)
        diff12 = (mu2 - mu1).reshape(-1, 1)
        diff21 = -diff12

        logdet_s1 = self._stable_logdet(s1e)
        logdet_s2 = self._stable_logdet(s2e)
        if not np.isfinite(logdet_s1) or not np.isfinite(logdet_s2):
            return float("inf")

        kl12 = 0.5 * (
            float(np.trace(inv_s2 @ s1e))
            + float((diff12.T @ inv_s2 @ diff12).item())
            - d
            + (logdet_s2 - logdet_s1)
        )
        kl21 = 0.5 * (
            float(np.trace(inv_s1 @ s2e))
            + float((diff21.T @ inv_s1 @ diff21).item())
            - d
            + (logdet_s1 - logdet_s2)
        )
        return float(max(0.5 * (kl12 + kl21), 0.0))

    def _pair_score(self, b1: GranularBall, b2: GranularBall) -> tuple[float, float]:
        if b1.covariance is None or b2.covariance is None:
            return float("inf"), 0.0
        mu1 = np.asarray(b1.center, dtype=np.float64)
        mu2 = np.asarray(b2.center, dtype=np.float64)
        s1 = np.asarray(b1.covariance, dtype=np.float64)
        s2 = np.asarray(b2.covariance, dtype=np.float64)
        try:
            if self.cfg.collision_metric == "sym_kl":
                dist = self._sym_kl_distance(mu1, s1, mu2, s2)
            else:
                dist = self._bhattacharyya_distance(mu1, s1, mu2, s2)
            score = float(np.exp(-dist)) if np.isfinite(dist) else 0.0
            return dist, score
        except np.linalg.LinAlgError:
            return float("inf"), 0.0

    def _distances(self, features: np.ndarray, balls: List[GranularBall]) -> np.ndarray:
        n = features.shape[0]
        m = len(balls)
        out = np.full((n, m), fill_value=np.inf, dtype=np.float64)
        for j, b in enumerate(balls):
            center = np.asarray(b.center, dtype=np.float64)
            delta = features - center[None, :]
            if b.inv_covariance is not None:
                inv_cov = np.asarray(b.inv_covariance, dtype=np.float64)
                out[:, j] = np.einsum("nd,dd,nd->n", delta, inv_cov, delta)
            else:
                out[:, j] = np.sum(delta * delta, axis=1)
        return out

    def decide(self, out: DirichletOutput, features=None, balls=None) -> ThreeWayOutput:
        if out.alpha.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=out.alpha.device)
            return ThreeWayOutput(empty, empty.bool(), empty.bool())

        if features is None or balls is None or len(balls) == 0:
            self.last_collision_pairs = []
            return self._legacy(out)

        x = np.asarray(features, dtype=np.float64)
        probs = out.alpha / torch.clamp(out.alpha.sum(dim=-1, keepdim=True), min=1e-8)
        max_prob = torch.max(probs, dim=-1).values
        uncertainty = out.uncertainty

        d2 = self._distances(x, balls)
        nearest_idx_np = np.argmin(d2, axis=1).astype(np.int64)
        nearest_d2_np = d2[np.arange(d2.shape[0]), nearest_idx_np]
        thresholds = np.array([self._ball_threshold(b) for b in balls], dtype=np.float64)
        inside = d2 <= thresholds[None, :]
        inside_any = torch.from_numpy(np.any(inside, axis=1)).to(out.alpha.device)

        collision_mask_np = np.zeros((x.shape[0],), dtype=bool)
        pairs: List[dict[str, Any]] = []
        per_ball_score = np.zeros((len(balls),), dtype=np.float64)
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                li = balls[i].dominant_class
                lj = balls[j].dominant_class
                if li is None or lj is None or li == lj:
                    continue
                dist, score = self._pair_score(balls[i], balls[j])
                if score < self.cfg.collision_threshold:
                    continue
                # Strong overlap should expand BND much faster than weak overlap.
                margin = self.cfg.boundary_scale_k1 + self.cfg.boundary_scale_k2 * (score / max(1e-6, 1.0 - score))
                margin = float(np.clip(margin, 0.0, 10.0))
                ti = thresholds[i] * (1.0 + margin)
                tj = thresholds[j] * (1.0 + margin)
                pair_mask = (d2[:, i] <= ti) & (d2[:, j] <= tj)
                collision_mask_np |= pair_mask
                per_ball_score[i] = max(per_ball_score[i], score)
                per_ball_score[j] = max(per_ball_score[j], score)
                pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "class_i": int(li),
                        "class_j": int(lj),
                        "distance": float(dist),
                        "score": float(score),
                        "margin": float(margin),
                        "metric": self.cfg.collision_metric,
                    }
                )
        self.last_collision_pairs = pairs
        for idx, b in enumerate(balls):
            b.boundary_score = float(per_ball_score[idx])

        collision_mask = torch.from_numpy(collision_mask_np).to(out.alpha.device)
        labels = torch.full_like(max_prob, fill_value=REGION_BOUNDARY, dtype=torch.long)

        accept = (max_prob >= self.cfg.alpha) & (uncertainty <= self.cfg.uncertainty_accept) & inside_any
        reject = (max_prob <= self.cfg.beta) | (uncertainty >= self.cfg.uncertainty_reject) | (~inside_any)

        labels[accept] = REGION_POSITIVE
        labels[reject] = REGION_NEGATIVE
        labels[collision_mask] = REGION_BOUNDARY

        nearest_idx = torch.from_numpy(nearest_idx_np).to(out.alpha.device)
        nearest_d2 = torch.from_numpy(nearest_d2_np.astype(np.float32)).to(out.alpha.device)
        return ThreeWayOutput(
            region_labels=labels,
            accept_mask=(labels == REGION_POSITIVE),
            reject_mask=(labels == REGION_NEGATIVE),
            collision_mask=collision_mask,
            nearest_ball_index=nearest_idx,
            mahalanobis_score=nearest_d2,
        )
