from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class FeatureBatch:
    features: torch.Tensor
    boxes: List[torch.Tensor]
    class_targets: List[torch.Tensor]
    image_ids: List[str]


@dataclass
class DirichletOutput:
    alpha: torch.Tensor
    evidence: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class GranularBall:
    center: np.ndarray
    radius: float
    members: List[int]
    purity: float
    topo_state: Dict[str, float | int | bool] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    inv_covariance: Optional[np.ndarray] = None
    chi2_threshold: Optional[float] = None
    semantic_entropy: Optional[float] = None
    dominant_class: Optional[int] = None
    confidence_level: Optional[float] = None
    boundary_score: Optional[float] = None
    geometry_type: Optional[str] = None
    support_count: Optional[int] = None


@dataclass
class ThreeWayOutput:
    region_labels: torch.Tensor
    accept_mask: torch.Tensor
    reject_mask: torch.Tensor
    collision_mask: Optional[torch.Tensor] = None
    nearest_ball_index: Optional[torch.Tensor] = None
    mahalanobis_score: Optional[torch.Tensor] = None


@dataclass
class VSFReport:
    gtf_auc: float
    snh_auc: float
    vsf_auc: float
    k_curve: List[Dict[str, float]]
    semantic_weight: float
    ks: List[int]
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class StepLosses:
    total: torch.Tensor
    detection: torch.Tensor
    edl: torch.Tensor
    gw: torch.Tensor
    topology: torch.Tensor
    hard_negative: torch.Tensor


def to_device(t: torch.Tensor | List[torch.Tensor], device: torch.device) -> torch.Tensor | List[torch.Tensor]:
    if isinstance(t, list):
        return [x.to(device) for x in t]
    return t.to(device)


REGION_NEGATIVE = 0
REGION_BOUNDARY = 1
REGION_POSITIVE = 2
