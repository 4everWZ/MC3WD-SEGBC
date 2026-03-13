from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from tegb.types import VSFReport


def _continuity(high: np.ndarray, low: np.ndarray, k: int) -> float:
    n = high.shape[0]
    d_high = np.linalg.norm(high[:, None, :] - high[None, :, :], axis=-1)
    d_low = np.linalg.norm(low[:, None, :] - low[None, :, :], axis=-1)
    r_high = np.argsort(np.argsort(d_high, axis=1), axis=1)
    r_low = np.argsort(np.argsort(d_low, axis=1), axis=1)

    penalty = 0.0
    for i in range(n):
        high_nn = set(np.argsort(d_high[i])[1 : k + 1].tolist())
        low_nn = set(np.argsort(d_low[i])[1 : k + 1].tolist())
        v_i = high_nn - low_nn
        for j in v_i:
            penalty += (r_high[i, j] - k)
    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return 0.0
    c = 1.0 - (2.0 / denom) * penalty
    return float(np.clip(c, 0.0, 1.0))


def _snh(low: np.ndarray, probs: np.ndarray, k: int) -> float:
    if probs.size == 0:
        return 0.0
    n = low.shape[0]
    d_low = np.linalg.norm(low[:, None, :] - low[None, :, :], axis=-1)
    sim = cosine_similarity(probs)
    vals = []
    for i in range(n):
        nn = np.argsort(d_low[i])[1 : k + 1]
        if nn.size == 0:
            continue
        vals.append(float(np.mean(sim[i, nn])))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _auc(ks: List[int], vals: List[float]) -> float:
    if len(ks) <= 1:
        return float(vals[0]) if vals else 0.0
    x = np.asarray(ks, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y=y, x=x)  # NumPy >= 2.0
    else:
        area = np.trapz(y=y, x=x)  # pragma: no cover
    span = max(1e-8, x[-1] - x[0])
    return float(np.clip(area / span, 0.0, 1.0))


def compute_vsf_report(
    high: np.ndarray,
    low: np.ndarray,
    probs: np.ndarray,
    semantic_weight: float = 0.5,
    ks: Iterable[int] | None = None,
) -> VSFReport:
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low sample count mismatch")
    n = high.shape[0]
    if n < 3:
        return VSFReport(0.0, 0.0, 0.0, [], semantic_weight, [])

    trust_k_max = max(1, int(np.ceil(n / 2.0) - 1))
    if ks is None:
        k_max = min(30, n - 1, trust_k_max)
        ks = [k for k in range(3, k_max + 1, 3)]
    ks = [k for k in ks if 0 < k < n and k <= trust_k_max]
    if not ks:
        ks = [max(1, min(3, n - 1, trust_k_max))]

    curve = []
    gtf_vals = []
    snh_vals = []
    vsf_vals = []

    for k in ks:
        t = float(np.clip(trustworthiness(high, low, n_neighbors=k), 0.0, 1.0))
        c = _continuity(high, low, k)
        gtf = 0.0 if (t + c) <= 1e-8 else float((2.0 * t * c) / (t + c))
        s = _snh(low, probs, k)
        vsf = float(np.clip(semantic_weight * gtf + (1.0 - semantic_weight) * s, 0.0, 1.0))
        curve.append({"k": float(k), "trustworthiness": t, "continuity": c, "gtf": gtf, "snh": s, "vsf": vsf})
        gtf_vals.append(gtf)
        snh_vals.append(s)
        vsf_vals.append(vsf)

    return VSFReport(
        gtf_auc=_auc(list(ks), gtf_vals),
        snh_auc=_auc(list(ks), snh_vals),
        vsf_auc=_auc(list(ks), vsf_vals),
        k_curve=curve,
        semantic_weight=semantic_weight,
        ks=list(ks),
    )


def compute_fpr95(in_scores: np.ndarray, ood_scores: np.ndarray) -> float | None:
    if in_scores.size == 0 or ood_scores.size == 0:
        return None
    thr = np.percentile(in_scores, 5.0)
    return float(np.mean(ood_scores >= thr))


def compute_coverage_at_risk(correct_mask: np.ndarray, confidence: np.ndarray, risk_threshold: float = 0.05) -> float | None:
    if correct_mask.size == 0 or confidence.size == 0:
        return None
    order = np.argsort(-confidence)
    correct = correct_mask[order].astype(np.float64)
    err = 1.0 - correct
    cum_risk = np.cumsum(err) / (np.arange(err.shape[0]) + 1.0)
    ok = np.where(cum_risk <= risk_threshold)[0]
    if ok.size == 0:
        return 0.0
    return float((ok.max() + 1) / err.shape[0])


def compute_cluster_indices(points: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    uniq = np.unique(labels)
    valid = uniq[uniq >= 0]
    if points.shape[0] < 3 or valid.size < 2:
        return {}
    out: Dict[str, float] = {}
    try:
        out["silhouette"] = float(silhouette_score(points, labels))
    except Exception:
        pass
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(points, labels))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(points, labels))
    except Exception:
        pass
    return out
