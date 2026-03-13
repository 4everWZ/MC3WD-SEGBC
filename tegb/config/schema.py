from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ExperimentSection:
    exp_name: str = "tegb_experiment"
    output_root: str = "runs"
    seed: int = 42
    device: str = "cuda"
    model_family: str = "yolo11"
    weights: str = "yolo11n.pt"
    target_layer: str = "model.model.22"
    num_classes: int = 80
    checkpoint_every: int = 1
    save_best: bool = True
    best_metric: str = "val_total"
    best_mode: str = "min"

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1:
            raise ValueError("experiment.checkpoint_every must be >= 1")
        if self.best_mode not in {"max", "min"}:
            raise ValueError("experiment.best_mode must be 'max' or 'min'")
        if not str(self.best_metric).strip():
            raise ValueError("experiment.best_metric must be a non-empty string")


@dataclass
class DataSection:
    images_dir: str = "datasets/coco128/images/train2017"
    labels_dir: str = "datasets/coco128/labels/train2017"
    image_size: int = 640
    crop_size: int = 224
    max_samples: int = 512
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    val_split_ratio: float = 0.2
    use_weighted_sampler: bool = False
    sampler_power: float = 1.0
    sampler_empty_weight: float = 0.2
    sampler_min_weight: float = 0.05
    sampler_max_weight: float = 20.0

    def __post_init__(self) -> None:
        if self.max_samples < 0:
            raise ValueError("data.max_samples must be >= 0")
        if self.batch_size < 1:
            raise ValueError("data.batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0")
        if not (0.0 <= self.val_split_ratio < 1.0):
            raise ValueError("data.val_split_ratio must be in [0, 1)")
        if self.sampler_power < 0.0:
            raise ValueError("data.sampler_power must be >= 0")
        if self.sampler_empty_weight <= 0.0:
            raise ValueError("data.sampler_empty_weight must be > 0")
        if self.sampler_min_weight <= 0.0:
            raise ValueError("data.sampler_min_weight must be > 0")
        if self.sampler_max_weight < self.sampler_min_weight:
            raise ValueError("data.sampler_max_weight must be >= data.sampler_min_weight")


@dataclass
class OptimSection:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    train_backbone: bool = False
    amp: bool = False


@dataclass
class LossSection:
    lambda_edl: float = 1.0
    lambda_gw: float = 0.2
    lambda_ph: float = 0.2
    lambda_hn: float = 0.1
    gw_entropy_reg: float = 5e-2
    gw_metric_weight: float = 0.1
    topo_max_points: int = 128
    enable_tgb: bool = True
    enable_edl: bool = True
    enable_gw: bool = True
    enable_ph: bool = True
    enable_3wd: bool = True
    enable_class_balanced_det: bool = True
    det_class_balance_power: float = 0.5
    det_class_weight_min: float = 0.25
    det_class_weight_max: float = 4.0

    def __post_init__(self) -> None:
        if self.det_class_balance_power < 0.0:
            raise ValueError("loss.det_class_balance_power must be >= 0")
        if self.det_class_weight_min <= 0.0:
            raise ValueError("loss.det_class_weight_min must be > 0")
        if self.det_class_weight_max < self.det_class_weight_min:
            raise ValueError("loss.det_class_weight_max must be >= loss.det_class_weight_min")


@dataclass
class GranularSection:
    mode: str = "v3_spherical_entropy"
    purity_threshold: float = 0.9
    min_members: int = 8
    max_balls: int = 256
    split_k: int = 2
    ph_persistence_threshold: float = 0.05
    radius_shrink_factor: float = 0.75
    confidence_level: float = 0.95
    entropy_threshold: float = 1.2
    radius_policy: str = "percentile"
    radius_percentile: float = 95.0
    entropy_log_base: float = 2.0
    cov_shrinkage: float = 0.2
    cov_jitter: float = 1e-4
    min_eig: float = 1e-6
    split_algorithm: str = "euclidean_kmeans"
    covariance_estimator: str = "ledoit_wolf"
    split_max_iter: int = 25
    split_tolerance: float = 1e-4
    split_min_child: int = 4
    max_split_depth: int = 12
    min_entropy_gain: float = 1e-3
    ph_max_points: int = 128

    def __post_init__(self) -> None:
        if self.mode not in {"v3_spherical_entropy", "prob_ellipsoid", "legacy_sphere"}:
            raise ValueError(f"Unsupported granular.mode: {self.mode}")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("granular.confidence_level must be in (0, 1)")
        if self.radius_policy not in {"percentile", "max"}:
            raise ValueError("granular.radius_policy must be 'percentile' or 'max'")
        if not (0.0 < self.radius_percentile <= 100.0):
            raise ValueError("granular.radius_percentile must be in (0, 100]")
        if self.entropy_log_base <= 0.0 or abs(self.entropy_log_base - 1.0) < 1e-12:
            raise ValueError("granular.entropy_log_base must be > 0 and != 1")
        if self.split_k < 2:
            raise ValueError("granular.split_k must be >= 2")
        if self.min_members < 1:
            raise ValueError("granular.min_members must be >= 1")
        if self.max_balls < 1:
            raise ValueError("granular.max_balls must be >= 1")
        if not (0.0 <= self.cov_shrinkage <= 1.0):
            raise ValueError("granular.cov_shrinkage must be in [0, 1]")
        if self.cov_jitter <= 0.0:
            raise ValueError("granular.cov_jitter must be > 0")
        if self.min_eig <= 0.0:
            raise ValueError("granular.min_eig must be > 0")
        if self.split_algorithm not in {"euclidean_kmeans", "mahalanobis_kmeans"}:
            raise ValueError(f"Unsupported granular.split_algorithm: {self.split_algorithm}")
        if self.mode == "v3_spherical_entropy" and self.split_algorithm != "euclidean_kmeans":
            raise ValueError("granular.split_algorithm must be 'euclidean_kmeans' when mode is v3_spherical_entropy")
        if self.covariance_estimator not in {"empirical", "ledoit_wolf", "oas"}:
            raise ValueError(f"Unsupported granular.covariance_estimator: {self.covariance_estimator}")
        if self.split_max_iter < 1:
            raise ValueError("granular.split_max_iter must be >= 1")
        if self.split_tolerance <= 0.0:
            raise ValueError("granular.split_tolerance must be > 0")
        if self.split_min_child < 1:
            raise ValueError("granular.split_min_child must be >= 1")
        if self.max_split_depth < 1:
            raise ValueError("granular.max_split_depth must be >= 1")
        if self.min_entropy_gain < 0.0:
            raise ValueError("granular.min_entropy_gain must be >= 0")
        if self.ph_max_points < 8:
            raise ValueError("granular.ph_max_points must be >= 8")


@dataclass
class DecisionSection:
    mode: str = "v3_spherical_collision"
    alpha: float = 0.9
    beta: float = 0.1
    uncertainty_accept: float = 0.2
    uncertainty_reject: float = 0.6
    collision_gamma: float = 1.0
    collision_metric: str = "bhattacharyya"
    collision_threshold: float = 0.8
    boundary_scale_k1: float = 0.2
    boundary_scale_k2: float = 0.8

    def __post_init__(self) -> None:
        if self.mode not in {"v3_spherical_collision", "hybrid_collision", "legacy_threshold"}:
            raise ValueError(f"Unsupported decision.mode: {self.mode}")
        if self.collision_gamma <= 0.0:
            raise ValueError("decision.collision_gamma must be > 0")
        if self.collision_metric not in {"bhattacharyya", "sym_kl"}:
            raise ValueError(f"Unsupported decision.collision_metric: {self.collision_metric}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("decision.alpha must be in [0, 1]")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("decision.beta must be in [0, 1]")
        if self.alpha < self.beta:
            raise ValueError("decision.alpha must be >= decision.beta")
        if self.uncertainty_accept < 0.0 or self.uncertainty_reject < 0.0:
            raise ValueError("decision uncertainty thresholds must be >= 0")
        if self.uncertainty_accept > self.uncertainty_reject:
            raise ValueError("decision.uncertainty_accept must be <= decision.uncertainty_reject")
        if not (0.0 <= self.collision_threshold <= 1.0):
            raise ValueError("decision.collision_threshold must be in [0, 1]")
        if self.boundary_scale_k1 < 0.0 or self.boundary_scale_k2 < 0.0:
            raise ValueError("decision.boundary_scale_k1/k2 must be >= 0")


@dataclass
class VSFSection:
    semantic_weight: float = 0.5
    k_min: int = 3
    k_max: int = 30
    k_step: int = 3
    max_points: int = 2000
    cluster_max_points: int = 5000
    diagnostics_max_points: int = 2500
    diagnostics_knn_k: int = 15
    diagnostics_min_ball_members: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.semantic_weight <= 1.0):
            raise ValueError("vsf.semantic_weight must be in [0, 1]")
        if self.k_min < 1:
            raise ValueError("vsf.k_min must be >= 1")
        if self.k_max < self.k_min:
            raise ValueError("vsf.k_max must be >= vsf.k_min")
        if self.k_step < 1:
            raise ValueError("vsf.k_step must be >= 1")
        if self.max_points < 0:
            raise ValueError("vsf.max_points must be >= 0")
        if self.cluster_max_points < 0:
            raise ValueError("vsf.cluster_max_points must be >= 0")
        if self.diagnostics_max_points < 0:
            raise ValueError("vsf.diagnostics_max_points must be >= 0")
        if self.diagnostics_knn_k < 1:
            raise ValueError("vsf.diagnostics_knn_k must be >= 1")
        if self.diagnostics_min_ball_members < 2:
            raise ValueError("vsf.diagnostics_min_ball_members must be >= 2")


@dataclass
class VisualizationSection:
    enabled: bool = True
    max_points: int = 6000
    embedding_random_subset_per_class: int = 0
    max_ball_overlays: int = 120
    class_legend_max_items: int = 14
    show_zero_count_classes_in_legend: bool = False
    focus_label_source: str = "target"
    group_cmap_name: str = "tab10"
    class_groups: List[List[str]] = field(default_factory=list)
    include_classes: List[str] = field(default_factory=list)
    class_compare_mode: str = "class"
    class_compare_topk: int = 2
    class_compare_label_source: str = "target"
    embedding_source: str = "head"
    embedding_compare_sources: List[str] = field(default_factory=list)
    embedding_auto_tsne_max_points: int = 10000
    embedding_metric: str = "cosine"
    highdim_pre_reduce: str = "pca"
    highdim_pca_components: int = 50
    tsne_perplexity: float = 35.0
    tsne_max_iter: int = 1000
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    datamap_enabled: bool = True
    datamap_force_matplotlib: bool = True
    datamap_use_medoids: bool = True
    datamap_point_size: int = 6
    datamap_alpha: float = 0.75
    datamap_min_font_size: int = 12
    datamap_max_font_size: int = 64
    datamap_label_wrap_width: int = 16
    datamap_title_font_size: int = 26
    embedding_scatter_alpha: float = 0.58
    embedding_color_lighten: float = 0.35
    embedding_pred_dominant_ratio_threshold: float = 0.7
    embedding_pred_dominant_alpha: float = 0.5
    balls_3d_enabled: bool = True
    balls_3d_radius_quantiles: List[float] = field(default_factory=lambda: [0.33, 0.66])
    ball_label_source: str = "target"

    def __post_init__(self) -> None:
        valid_embedding_sources = {"head", "auto_high", "high_pca", "high_tsne", "high_umap"}
        if self.max_points < 1:
            raise ValueError("visualization.max_points must be >= 1")
        if self.embedding_random_subset_per_class < 0:
            raise ValueError("visualization.embedding_random_subset_per_class must be >= 0")
        if self.max_ball_overlays < 1:
            raise ValueError("visualization.max_ball_overlays must be >= 1")
        if self.class_legend_max_items < 1:
            raise ValueError("visualization.class_legend_max_items must be >= 1")
        if self.focus_label_source not in {"target", "pred"}:
            raise ValueError("visualization.focus_label_source must be 'target' or 'pred'")
        if self.class_compare_mode not in {"class", "topk"}:
            raise ValueError("visualization.class_compare_mode must be 'class' or 'topk'")
        if self.class_compare_topk < 1:
            raise ValueError("visualization.class_compare_topk must be >= 1")
        if self.class_compare_label_source not in {"target", "pred"}:
            raise ValueError("visualization.class_compare_label_source must be 'target' or 'pred'")
        if self.embedding_source not in valid_embedding_sources:
            raise ValueError("visualization.embedding_source must be one of head/auto_high/high_pca/high_tsne/high_umap")
        normalized_compare_sources: List[str] = []
        seen_compare_sources: set[str] = set()
        for token in self.embedding_compare_sources:
            source = str(token).strip()
            if not source or source in seen_compare_sources:
                continue
            if source not in valid_embedding_sources:
                raise ValueError(
                    "visualization.embedding_compare_sources must contain only head/auto_high/high_pca/high_tsne/high_umap"
                )
            normalized_compare_sources.append(source)
            seen_compare_sources.add(source)
        self.embedding_compare_sources = normalized_compare_sources
        if self.embedding_auto_tsne_max_points < 2:
            raise ValueError("visualization.embedding_auto_tsne_max_points must be >= 2")
        if self.highdim_pre_reduce not in {"none", "pca"}:
            raise ValueError("visualization.highdim_pre_reduce must be 'none' or 'pca'")
        if self.highdim_pca_components < 2:
            raise ValueError("visualization.highdim_pca_components must be >= 2")
        if self.tsne_perplexity <= 0.0:
            raise ValueError("visualization.tsne_perplexity must be > 0")
        if self.tsne_max_iter < 250:
            raise ValueError("visualization.tsne_max_iter must be >= 250")
        if self.umap_n_neighbors < 2:
            raise ValueError("visualization.umap_n_neighbors must be >= 2")
        if not (0.0 <= self.umap_min_dist <= 1.0):
            raise ValueError("visualization.umap_min_dist must be in [0, 1]")
        if self.datamap_point_size < 1:
            raise ValueError("visualization.datamap_point_size must be >= 1")
        if not (0.0 < self.datamap_alpha <= 1.0):
            raise ValueError("visualization.datamap_alpha must be in (0, 1]")
        if self.datamap_min_font_size < 1:
            raise ValueError("visualization.datamap_min_font_size must be >= 1")
        if self.datamap_max_font_size < self.datamap_min_font_size:
            raise ValueError(
                "visualization.datamap_max_font_size must be >= visualization.datamap_min_font_size"
            )
        if self.datamap_label_wrap_width < 4:
            raise ValueError("visualization.datamap_label_wrap_width must be >= 4")
        if self.datamap_title_font_size < 8:
            raise ValueError("visualization.datamap_title_font_size must be >= 8")
        if not (0.0 < self.embedding_scatter_alpha <= 1.0):
            raise ValueError("visualization.embedding_scatter_alpha must be in (0, 1]")
        if not (0.0 <= self.embedding_color_lighten <= 1.0):
            raise ValueError("visualization.embedding_color_lighten must be in [0, 1]")
        if not (0.0 <= self.embedding_pred_dominant_ratio_threshold <= 1.0):
            raise ValueError("visualization.embedding_pred_dominant_ratio_threshold must be in [0, 1]")
        if not (0.0 < self.embedding_pred_dominant_alpha <= 1.0):
            raise ValueError("visualization.embedding_pred_dominant_alpha must be in (0, 1]")
        if (
            not isinstance(self.balls_3d_radius_quantiles, list)
            or len(self.balls_3d_radius_quantiles) != 2
        ):
            raise ValueError("visualization.balls_3d_radius_quantiles must be a list of length 2")
        q1 = float(self.balls_3d_radius_quantiles[0])
        q2 = float(self.balls_3d_radius_quantiles[1])
        if not (0.0 < q1 < q2 < 1.0):
            raise ValueError("visualization.balls_3d_radius_quantiles must satisfy 0 < q1 < q2 < 1")
        self.balls_3d_radius_quantiles = [q1, q2]
        if self.ball_label_source not in {"target", "pred", "dominant"}:
            raise ValueError("visualization.ball_label_source must be one of target/pred/dominant")
        normalized_groups: List[List[str]] = []
        for group in self.class_groups:
            if not isinstance(group, list):
                continue
            cleaned = [str(x).strip() for x in group if str(x).strip()]
            if cleaned:
                normalized_groups.append(cleaned)
        self.class_groups = normalized_groups
        self.include_classes = [str(x).strip() for x in self.include_classes if str(x).strip()]


@dataclass
class AblationSection:
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    variants: List[str] = field(
        default_factory=lambda: ["Full", "-TGB", "-EDL", "-GW", "-PH", "-3WD", "-MAXR"]
    )


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    optim: OptimSection = field(default_factory=OptimSection)
    loss: LossSection = field(default_factory=LossSection)
    granular: GranularSection = field(default_factory=GranularSection)
    decision: DecisionSection = field(default_factory=DecisionSection)
    vsf: VSFSection = field(default_factory=VSFSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)
    ablation: AblationSection = field(default_factory=AblationSection)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_root) / self.experiment.exp_name


def _deep_update(base: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in custom.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(cfg_dict: Dict[str, Any]) -> ExperimentConfig:
    vis_cfg = dict(cfg_dict.get("visualization", {}))
    # Backward compatibility: topk_classes was removed from visualization schema.
    vis_cfg.pop("topk_classes", None)
    # Backward compatibility: embedding_balance_mode was removed; visualization uses cap-only sampling.
    vis_cfg.pop("embedding_balance_mode", None)
    return ExperimentConfig(
        experiment=ExperimentSection(**cfg_dict.get("experiment", {})),
        data=DataSection(**cfg_dict.get("data", {})),
        optim=OptimSection(**cfg_dict.get("optim", {})),
        loss=LossSection(**cfg_dict.get("loss", {})),
        granular=GranularSection(**cfg_dict.get("granular", {})),
        decision=DecisionSection(**cfg_dict.get("decision", {})),
        vsf=VSFSection(**cfg_dict.get("vsf", {})),
        visualization=VisualizationSection(**vis_cfg),
        ablation=AblationSection(**cfg_dict.get("ablation", {})),
    )


def load_config(path: str) -> ExperimentConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        custom = yaml.safe_load(f) or {}
    base = ExperimentConfig().to_dict()
    merged = _deep_update(base, custom)
    return _coerce(merged)
