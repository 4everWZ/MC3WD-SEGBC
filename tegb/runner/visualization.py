from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

_DEFAULT_CLASS_LABEL_ALIASES: Dict[str, str] = {
    "traffic light": "t-light",
    "fire hydrant": "hydrant",
    "stop sign": "stop",
    "parking meter": "meter",
    "sports ball": "ball",
    "baseball bat": "bat",
    "baseball glove": "glove",
    "tennis racket": "racket",
    "wine glass": "glass",
    "dining table": "table",
    "potted plant": "plant",
    "cell phone": "phone",
    "hair drier": "dryer",
    "hot dog": "hotdog",
    "teddy bear": "teddy",
}

_VALID_EMBEDDING_SOURCES = {"head", "auto_high", "high_pca", "high_tsne", "high_umap"}


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return np.sort(idx.astype(np.int64))


def _class_capped_indices(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    max_per_class: int,
    seed: int,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return candidate_indices.astype(np.int64, copy=False)
    if max_per_class <= 0:
        return np.sort(candidate_indices.astype(np.int64, copy=False))

    lbl = labels[candidate_indices].astype(np.int64, copy=False)
    classes = np.unique(lbl)
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for cls in classes.tolist():
        cls_mask = lbl == int(cls)
        cls_idx = candidate_indices[cls_mask]
        if cls_idx.size <= int(max_per_class):
            chunks.append(np.sort(cls_idx.astype(np.int64, copy=False)))
            continue
        sampled = rng.choice(cls_idx, size=int(max_per_class), replace=False)
        chunks.append(np.sort(sampled.astype(np.int64, copy=False)))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.sort(np.concatenate(chunks).astype(np.int64, copy=False))


def _label_name(v: int) -> str:
    if v == 0:
        return "NEG"
    if v == 1:
        return "BND"
    if v == 2:
        return "POS"
    return str(v)


def _normalize_class_name_map(
    class_names: Mapping[int, str] | Sequence[str] | None,
    labels: Sequence[np.ndarray],
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if isinstance(class_names, Mapping):
        for k, v in class_names.items():
            try:
                mapping[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(class_names, (list, tuple)):
        for i, v in enumerate(class_names):
            mapping[i] = str(v)

    max_id = -1
    for arr in labels:
        if arr is not None and arr.size > 0:
            max_id = max(max_id, int(np.max(arr)))
    for i in range(max_id + 1):
        mapping.setdefault(i, f"class_{i}")
    return mapping


def _name_key(token: Any) -> str:
    text = str(token).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _resolve_class_id(token: Any, class_name_to_id: Mapping[str, int], class_map: Mapping[int, str]) -> int | None:
    if isinstance(token, (int, np.integer)):
        cid = int(token)
        return cid if cid in class_map else None
    text = str(token).strip()
    if not text:
        return None
    if text.isdigit():
        cid = int(text)
        return cid if cid in class_map else None
    return class_name_to_id.get(_name_key(text))


def _resolve_focus_ids(
    include_classes: Sequence[Any] | None,
    class_map: Mapping[int, str],
) -> tuple[set[int], list[str]]:
    if not include_classes:
        return set(), []
    class_name_to_id: Dict[str, int] = {}
    for k, v in class_map.items():
        cid = int(k)
        class_name_to_id[_name_key(v)] = cid
    out: set[int] = set()
    unresolved: list[str] = []
    for token in include_classes:
        cid = _resolve_class_id(token, class_name_to_id, class_map)
        if cid is not None:
            out.add(cid)
        else:
            unresolved.append(str(token))
    return out, unresolved


def _build_display_class_map(class_map: Mapping[int, str]) -> Dict[int, str]:
    display_map: Dict[int, str] = {}
    for cid, name in class_map.items():
        key = _name_key(name)
        display_map[int(cid)] = _DEFAULT_CLASS_LABEL_ALIASES.get(key, str(name))
    return display_map


def _build_grouped_class_colors(
    present_class_ids: Sequence[int],
    class_map: Mapping[int, str],
    class_groups: Sequence[Sequence[Any]] | None,
    group_cmap_name: str,
    plt: Any,
) -> tuple[Dict[int, tuple[float, float, float, float]], Dict[int, int], list[dict[str, Any]], list[str]]:
    import matplotlib.colors as mcolors

    present = sorted({int(x) for x in present_class_ids if int(x) in class_map})
    if not present:
        return {}, {}, [], []

    class_name_to_id: Dict[str, int] = {}
    for k, v in class_map.items():
        cid = int(k)
        class_name_to_id[_name_key(v)] = cid
    class_to_group: Dict[int, int] = {}
    groups_present: list[list[int]] = []
    unresolved: list[str] = []

    if class_groups:
        for group in class_groups:
            resolved: list[int] = []
            for token in group:
                cid = _resolve_class_id(token, class_name_to_id, class_map)
                if cid is None:
                    unresolved.append(str(token))
                    continue
                if cid not in present or cid in resolved:
                    continue
                resolved.append(cid)
            if resolved:
                groups_present.append(resolved)
                gi = len(groups_present) - 1
                for cid in resolved:
                    class_to_group[cid] = gi
    explicit_group_count = len(groups_present)

    for cid in present:
        if cid not in class_to_group:
            groups_present.append([cid])
            class_to_group[cid] = len(groups_present) - 1

    try:
        base_cmap = plt.get_cmap(group_cmap_name)
    except Exception:
        base_cmap = plt.get_cmap("tab10")
    auto_cmap = plt.get_cmap("tab20")

    class_colors: Dict[int, tuple[float, float, float, float]] = {}
    group_rows: list[dict[str, Any]] = []
    g_total = len(groups_present)

    # Key rule for readability:
    # - configured groups: distinct hues across groups
    # - members inside one group: same hue, varied saturation/value
    listed = getattr(base_cmap, "colors", None)
    if listed is not None:
        discrete_colors = [mcolors.to_rgba(c) for c in list(listed)]
    else:
        discrete_colors = []
    if explicit_group_count > 0 and not discrete_colors:
        base_positions = np.linspace(0.0, 1.0, num=explicit_group_count, endpoint=False)
    else:
        base_positions = np.linspace(0.0, 1.0, num=max(1, g_total), endpoint=False)

    for gi, members in enumerate(groups_present):
        if gi < explicit_group_count:
            if discrete_colors:
                base_rgba = discrete_colors[gi % len(discrete_colors)]
            else:
                base_rgba = base_cmap(float(base_positions[gi]))
        else:
            auto_idx = gi - explicit_group_count
            base_rgba = auto_cmap((auto_idx % 20) / 19.0)
        hsv = mcolors.rgb_to_hsv(np.array([base_rgba[:3]], dtype=np.float64))[0]
        h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
        members_sorted = sorted(members, key=lambda x: class_map.get(int(x), str(x)))
        if len(members_sorted) <= 1:
            hsv_variants = [(h, min(1.0, max(0.35, s)), min(1.0, max(0.68, v)))]
        else:
            sat = np.linspace(min(1.0, max(0.40, s * 0.90)), min(1.0, max(0.62, s * 1.08)), num=len(members_sorted))
            val = np.linspace(0.62, 0.96, num=len(members_sorted))
            hsv_variants = [(h, float(sat[j]), float(val[j])) for j in range(len(members_sorted))]
        for j, cid in enumerate(members_sorted):
            hh, ss, vv = hsv_variants[j]
            rgb = mcolors.hsv_to_rgb(np.array([[hh, ss, vv]], dtype=np.float64))[0]
            class_colors[int(cid)] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        group_rows.append(
            {
                "group_index": int(gi),
                "member_ids": [int(c) for c in members_sorted],
                "member_names": [class_map.get(int(c), f"class_{int(c)}") for c in members_sorted],
                "from_config": bool(gi < explicit_group_count),
            }
        )

    return class_colors, class_to_group, group_rows, unresolved


def _lighten_rgba(color: Any, amount: float) -> tuple[float, float, float, float]:
    import matplotlib.colors as mcolors

    rgba = np.asarray(mcolors.to_rgba(color), dtype=np.float64).copy()
    amt = float(np.clip(amount, 0.0, 1.0))
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * amt
    rgba[:3] = np.clip(rgba[:3], 0.0, 1.0)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


def _lighten_color_map(
    class_colors: Mapping[int, tuple[float, float, float, float]],
    amount: float,
) -> Dict[int, tuple[float, float, float, float]]:
    return {int(k): _lighten_rgba(v, amount) for k, v in class_colors.items()}


def _pca_reduce(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    if x.ndim != 2 or x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    max_comp = min(x.shape[0] - 1, x.shape[1])
    if max_comp < 2:
        return np.pad(x[:, :1], ((0, 0), (0, 1)), mode="constant").astype(np.float32)
    n_comp = int(max(2, min(n_components, max_comp)))
    y = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
    return y.astype(np.float32, copy=False)


def _project_high_embedding(
    *,
    high: np.ndarray,
    low_fallback: np.ndarray,
    embedding_source: str,
    seed: int,
    auto_tsne_max_points: int,
    metric: str,
    pre_reduce: str,
    pca_components: int,
    tsne_perplexity: float,
    tsne_max_iter: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> tuple[np.ndarray, str, list[str]]:
    warnings: list[str] = []
    if embedding_source == "head":
        return low_fallback.astype(np.float32, copy=False), "head", warnings

    if high.ndim != 2 or high.shape[0] == 0:
        warnings.append("high_feature_unavailable_fallback_to_head")
        return low_fallback.astype(np.float32, copy=False), "head_fallback", warnings

    x = np.asarray(high, dtype=np.float32)
    if pre_reduce == "pca":
        try:
            x = _pca_reduce(x, n_components=pca_components, seed=seed)
        except Exception:
            warnings.append("pca_prereduce_failed")

    mode = embedding_source
    if mode == "auto_high":
        mode = "high_tsne" if x.shape[0] <= int(auto_tsne_max_points) else "high_umap"

    if mode == "high_pca":
        y = _pca_reduce(x, n_components=2, seed=seed)
        return y.astype(np.float32, copy=False), "high_pca", warnings

    if x.shape[0] < 5:
        warnings.append("too_few_points_projection_fallback_pca2")
        y = _pca_reduce(x, n_components=2, seed=seed)
        return y.astype(np.float32, copy=False), "pca2_fallback", warnings

    if mode == "high_umap":
        try:
            import umap  # type: ignore

            n_neighbors = int(max(2, min(int(umap_n_neighbors), x.shape[0] - 1)))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=float(umap_min_dist),
                metric=str(metric),
                random_state=int(seed),
            )
            y = reducer.fit_transform(x)
            return np.asarray(y, dtype=np.float32), "high_umap", warnings
        except Exception:
            warnings.append("umap_unavailable_or_failed_fallback_tsne")
            mode = "high_tsne"

    # t-SNE path (requested or fallback).
    try:
        from sklearn.manifold import TSNE

        perp_max = max(5.0, float(x.shape[0] - 1.0))
        perp = float(max(5.0, min(float(tsne_perplexity), perp_max)))
        if perp >= x.shape[0]:
            perp = float(max(1.0, x.shape[0] - 1.0))

        try:
            model = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                max_iter=int(tsne_max_iter),
                random_state=int(seed),
            )
        except TypeError:
            # Older scikit-learn compatibility.
            model = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                n_iter=int(tsne_max_iter),
                random_state=int(seed),
            )
        y = model.fit_transform(x)
        return np.asarray(y, dtype=np.float32), "high_tsne", warnings
    except Exception:
        warnings.append("tsne_failed_fallback_pca2")
        y = _pca_reduce(x, n_components=2, seed=seed)
        return y.astype(np.float32, copy=False), "pca2_fallback", warnings


def _plot_embedding_by_labels(
    *,
    ax: Any,
    pts: np.ndarray,
    labels: np.ndarray,
    class_map: Mapping[int, str],
    class_colors: Mapping[int, tuple[float, float, float, float]],
    selected_ids: Sequence[int] | None,
    title: str,
    legend_max_items: int = 14,
    class_to_group: Mapping[int, int] | None = None,
    configured_group_count: int = 0,
    show_group_prefix: bool = False,
    show_missing_in_legend: bool = False,
    scatter_alpha: float = 0.82,
    alpha_overrides: Mapping[int, float] | None = None,
) -> list[int]:
    import matplotlib.pyplot as plt

    if labels.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        return []

    counts = Counter([int(x) for x in labels.tolist()])
    ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    if selected_ids is not None:
        provided: list[int] = []
        seen: set[int] = set()
        for x in selected_ids:
            cid = int(x)
            if cid in seen:
                continue
            provided.append(cid)
            seen.add(cid)
        selected = sorted(provided, key=lambda c: counts.get(int(c), 0), reverse=True)
    else:
        selected = ordered

    fallback_cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(selected):
        mask = labels == cid
        g_prefix = ""
        if show_group_prefix and class_to_group is not None:
            gidx = class_to_group.get(int(cid), -1)
            if 0 <= int(gidx) < int(configured_group_count):
                g_prefix = f"G{int(gidx) + 1}:"
        color = class_colors.get(int(cid), fallback_cmap(i / max(1, len(selected) - 1)))
        name = class_map.get(int(cid), f"class_{int(cid)}")
        label = f"{g_prefix}{name} ({int(counts.get(int(cid), 0))})"
        if not np.any(mask):
            if show_missing_in_legend:
                ax.scatter([], [], s=18, alpha=0.9, color=color, marker="x", linewidths=0, label=label)
            continue
        alpha_value = float(alpha_overrides.get(int(cid), scatter_alpha)) if alpha_overrides is not None else float(scatter_alpha)
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=12,
            alpha=alpha_value,
            color=color,
            linewidths=0,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    handles, labels_txt = ax.get_legend_handles_labels()
    if handles:
        keep = max(1, int(legend_max_items))
        ax.legend(handles[:keep], labels_txt[:keep], loc="best", fontsize=8, frameon=True)
    return selected


def _ordered_classes_by_count(labels: np.ndarray, selected_ids: Sequence[int] | None = None) -> list[int]:
    counts = Counter([int(x) for x in labels.tolist()])
    ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    if selected_ids is None:
        return ordered
    allow = {int(x) for x in selected_ids}
    return [c for c in ordered if c in allow]


def _group_prefix(
    cid: int,
    class_to_group: Mapping[int, int],
    configured_group_rows: Sequence[Mapping[str, Any]],
) -> str:
    gidx = int(class_to_group.get(int(cid), -1))
    if 0 <= gidx < len(configured_group_rows):
        return f"G{gidx + 1}:"
    return ""


def _group_label_name(
    cid: int,
    class_to_group: Mapping[int, int],
    configured_group_rows: Sequence[Mapping[str, Any]],
    class_map: Mapping[int, str],
) -> str:
    gidx = int(class_to_group.get(int(cid), -1))
    if 0 <= gidx < len(configured_group_rows):
        member_ids = [int(x) for x in configured_group_rows[gidx].get("member_ids", [])]
        names = [class_map.get(int(x), f"class_{int(x)}") for x in member_ids]
        if names:
            return f"G{gidx + 1}:" + " / ".join(names)
        return f"G{gidx + 1}:{class_map.get(int(cid), f'class_{int(cid)}')}"
    return class_map.get(int(cid), f"class_{int(cid)}")


def _build_datamap_labels(
    label_ids: np.ndarray,
    *,
    class_map: Mapping[int, str],
    class_to_group: Mapping[int, int],
    configured_group_rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> np.ndarray:
    out: list[str] = []
    for cid_raw in label_ids.tolist():
        cid = int(cid_raw)
        cname = class_map.get(cid, f"class_{cid}")
        if mode == "group":
            out.append(_group_label_name(cid, class_to_group, configured_group_rows, class_map))
        elif mode == "class_with_group":
            out.append(f"{_group_prefix(cid, class_to_group, configured_group_rows)}{cname}")
        else:
            out.append(cname)
    return np.asarray(out, dtype=object)


def _save_embedding_payload(
    path: Path,
    *,
    index: np.ndarray,
    projected_points: np.ndarray,
    head_points: np.ndarray,
    high_points: np.ndarray | None,
    preds: np.ndarray,
    targets: np.ndarray,
    regions: np.ndarray,
    uncertainty: np.ndarray,
    collision_mask: np.ndarray | None,
    embedding_source_used: str,
) -> None:
    np.savez_compressed(
        path,
        index=np.asarray(index, dtype=np.int64),
        projected_points=np.asarray(projected_points, dtype=np.float32),
        head_points=np.asarray(head_points, dtype=np.float32),
        high_points=(
            np.asarray(high_points, dtype=np.float32)
            if high_points is not None
            else np.zeros((head_points.shape[0], 0), dtype=np.float32)
        ),
        preds=np.asarray(preds, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        regions=np.asarray(regions, dtype=np.int64),
        uncertainty=np.asarray(uncertainty, dtype=np.float32),
        collision_mask=(
            np.asarray(collision_mask, dtype=bool)
            if collision_mask is not None
            else np.zeros((projected_points.shape[0],), dtype=bool)
        ),
        embedding_source_used=np.asarray([str(embedding_source_used)], dtype=object),
    )


def _export_datamapplot_figure(
    *,
    points: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
    point_size: int,
    alpha: float,
    min_font_size: int,
    max_font_size: int,
    label_wrap_width: int,
    title_font_size: int,
    force_matplotlib: bool,
    use_medoids: bool,
) -> tuple[bool, str | None]:
    try:
        import datamapplot  # type: ignore
    except Exception as exc:
        return False, f"datamapplot_unavailable:{exc.__class__.__name__}"

    kwargs: Dict[str, Any] = {
        "title": str(title),
        "label_over_points": True,
        "dynamic_label_size": True,
        "point_size": int(point_size),
        "alpha": float(alpha),
        "force_matplotlib": bool(force_matplotlib),
        "use_medoids": bool(use_medoids),
        "label_wrap_width": int(label_wrap_width),
        "max_font_size": int(max_font_size),
        "min_font_size": int(min_font_size),
        "font_family": "DejaVu Sans",
        "title_keywords": {"fontsize": int(title_font_size), "fontweight": "bold"},
    }

    try:
        fig, ax = datamapplot.create_plot(points, labels, **kwargs)
    except Exception as exc:
        return False, f"datamapplot_render_failed:{exc.__class__.__name__}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return True, None


def export_visualization_artifacts(
    run_dir: Path,
    low: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    regions: np.ndarray,
    uncertainty: np.ndarray,
    collision_mask: np.ndarray | None,
    balls: list[Any],
    *,
    high: np.ndarray | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    class_groups: Sequence[Sequence[Any]] | None = None,
    include_classes: Sequence[Any] | None = None,
    group_cmap_name: str = "tab10",
    class_legend_max_items: int = 14,
    show_zero_count_classes_in_legend: bool = False,
    focus_label_source: str = "target",
    class_compare_mode: str = "class",
    class_compare_topk: int = 2,
    class_compare_label_source: str = "target",
    embedding_source: str = "head",
    embedding_auto_tsne_max_points: int = 10000,
    embedding_metric: str = "cosine",
    highdim_pre_reduce: str = "pca",
    highdim_pca_components: int = 50,
    tsne_perplexity: float = 35.0,
    tsne_max_iter: int = 1000,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    datamap_enabled: bool = True,
    datamap_force_matplotlib: bool = True,
    datamap_use_medoids: bool = True,
    datamap_point_size: int = 6,
    datamap_alpha: float = 0.75,
    datamap_min_font_size: int = 8,
    datamap_max_font_size: int = 42,
    datamap_label_wrap_width: int = 16,
    datamap_title_font_size: int = 22,
    embedding_scatter_alpha: float = 0.58,
    embedding_color_lighten: float = 0.35,
    embedding_pred_dominant_ratio_threshold: float = 0.7,
    embedding_pred_dominant_alpha: float = 0.5,
    balls_3d_enabled: bool = True,
    balls_3d_radius_quantiles: Sequence[float] = (0.33, 0.66),
    seed: int = 42,
    max_points: int = 6000,
    embedding_random_subset_per_class: int = 0,
    ball_label_source: str = "target",
    max_ball_overlays: int = 120,
    artifact_subdir: str = "visualizations",
    embedding_payload_subpath: str = "embedding_payload.npz",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "files": [],
        "warnings": [],
        "num_points_total": int(low.shape[0]),
    }
    artifact_subdir_clean = str(artifact_subdir).strip().replace("\\", "/").strip("/") or "visualizations"
    payload_subpath_clean = str(embedding_payload_subpath).strip().replace("\\", "/").strip("/") or "embedding_payload.npz"
    vis_dir = run_dir / "artifacts" / Path(artifact_subdir_clean)
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
    except Exception:
        out["enabled"] = False
        out["warnings"].append("matplotlib_unavailable")
        return out

    if low.ndim != 2 or low.shape[0] == 0:
        out["enabled"] = False
        out["warnings"].append("no_embedding_points")
        return out

    n_all = int(low.shape[0])
    pred_all = preds if preds.shape[0] == n_all else np.zeros((n_all,), dtype=np.int64)
    tgt_all = targets if targets.shape[0] == n_all else np.zeros((n_all,), dtype=np.int64)

    compare_mode = str(class_compare_mode).strip().lower()
    if compare_mode not in {"class", "topk"}:
        compare_mode = "class"
        out["warnings"].append("invalid_class_compare_mode_fallback_class")
    compare_topk = max(1, int(class_compare_topk))
    compare_source = str(class_compare_label_source).strip().lower()
    if compare_source not in {"target", "pred"}:
        compare_source = "target"
        out["warnings"].append("invalid_class_compare_label_source_fallback_target")
    legend_limit = max(1, int(class_legend_max_items))
    ball_label_mode = str(ball_label_source).strip().lower()
    if ball_label_mode not in {"target", "pred", "dominant"}:
        ball_label_mode = "target"
        out["warnings"].append("invalid_ball_label_source_fallback_target")
    qvals = np.asarray(list(balls_3d_radius_quantiles), dtype=np.float64)
    if qvals.size != 2 or not (0.0 < qvals[0] < qvals[1] < 1.0):
        qvals = np.asarray([0.33, 0.66], dtype=np.float64)
        out["warnings"].append("invalid_balls_3d_radius_quantiles_fallback_default")

    raw_class_map = _normalize_class_name_map(class_names, [pred_all, tgt_all])
    display_class_map = _build_display_class_map(raw_class_map)
    include_ids, unresolved_include = _resolve_focus_ids(include_classes, raw_class_map)
    if include_classes and not include_ids:
        out["warnings"].append("include_classes_not_resolved")
    if unresolved_include:
        out["warnings"].append("include_classes_partially_unresolved")
        out["include_classes_unresolved"] = sorted(set(unresolved_include))
    present_ids = sorted({int(x) for x in np.unique(np.concatenate([pred_all, tgt_all]))})
    class_colors, class_to_group, group_rows, unresolved_groups = _build_grouped_class_colors(
        present_class_ids=present_ids,
        class_map=raw_class_map,
        class_groups=class_groups,
        group_cmap_name=group_cmap_name,
        plt=plt,
    )
    soft_class_colors = _lighten_color_map(class_colors, amount=float(embedding_color_lighten))
    configured_group_rows = [row for row in group_rows if bool(row.get("from_config", False))]
    configured_group_member_ids: set[int] = set()
    for row in configured_group_rows:
        for cid in row.get("member_ids", []):
            configured_group_member_ids.add(int(cid))
    if unresolved_groups:
        out["warnings"].append("class_groups_partially_unresolved")
        out["class_groups_unresolved"] = sorted(set(unresolved_groups))
    out["class_groups_resolved"] = [
        {
            **row,
            "member_names": [
                display_class_map.get(int(cid), f"class_{int(cid)}")
                for cid in row.get("member_ids", [])
            ],
        }
        for row in configured_group_rows
    ]
    out["class_groups_auto_singletons"] = int(max(0, len(group_rows) - len(configured_group_rows)))
    out["focus_class_ids"] = sorted(int(x) for x in include_ids)
    out["focus_class_names"] = [display_class_map.get(int(x), f"class_{int(x)}") for x in sorted(include_ids)]
    out["class_label_display_map"] = {str(k): str(v) for k, v in display_class_map.items()}

    def _select_from_source(src_labels: np.ndarray, strict_group: bool) -> list[int]:
        present = sorted({int(x) for x in np.unique(src_labels)})
        if include_ids:
            return [c for c in present if c in include_ids]
        if strict_group:
            selected = [c for c in present if c in configured_group_member_ids]
            return selected
        return present

    src_all = tgt_all if focus_label_source == "target" else pred_all
    selected_all = _select_from_source(src_all.astype(np.int64, copy=False), strict_group=bool(class_groups))
    if not selected_all:
        out["warnings"].append("no_points_after_class_filter_fallback_all_classes")
        selected_all = sorted({int(x) for x in np.unique(src_all)})
    selected_set_all = set(int(x) for x in selected_all)
    cand_idx = np.where(np.isin(src_all, np.array(sorted(selected_set_all), dtype=np.int64)))[0]
    if cand_idx.size == 0:
        cand_idx = np.arange(n_all, dtype=np.int64)
    out["num_points_after_class_filter"] = int(cand_idx.size)

    # Visualization-only cap sampling for readability. Core metrics always use full long-tail data.
    cand_idx = _class_capped_indices(
        labels=src_all.astype(np.int64, copy=False),
        candidate_indices=cand_idx.astype(np.int64, copy=False),
        max_per_class=int(embedding_random_subset_per_class),
        seed=int(seed),
    )
    if cand_idx.size == 0:
        out["warnings"].append("class_cap_sampling_empty_fallback_all_candidates")
        cand_idx = np.sort(np.asarray(np.where(np.isin(src_all, np.array(sorted(selected_set_all), dtype=np.int64)))[0], dtype=np.int64))
    out["embedding_random_subset_per_class"] = int(embedding_random_subset_per_class)
    out["num_points_after_class_cap"] = int(cand_idx.size)
    out["ball_label_source"] = str(ball_label_mode)
    out["show_zero_count_classes_in_legend"] = bool(show_zero_count_classes_in_legend)

    if cand_idx.size <= int(max_points):
        idx = np.sort(cand_idx.astype(np.int64, copy=False))
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(cand_idx, size=int(max_points), replace=False).astype(np.int64, copy=False))

    low_sub = low[idx]
    high_sub = (high[idx] if (high is not None and getattr(high, "shape", (0,))[0] == n_all) else None)
    pred_sub = pred_all[idx]
    tgt_sub = tgt_all[idx]
    reg_sub = regions[idx] if regions.shape[0] == n_all else np.ones((idx.shape[0],), dtype=np.int64)
    unc_sub = uncertainty[idx] if uncertainty.shape[0] == n_all else np.zeros((idx.shape[0],), dtype=np.float32)
    col_sub = collision_mask[idx] if (collision_mask is not None and collision_mask.shape[0] == n_all) else None
    out["num_points_plotted"] = int(idx.shape[0])
    out["pred_unique_classes_plotted"] = int(np.unique(pred_sub).size)
    out["target_unique_classes_plotted"] = int(np.unique(tgt_sub).size)

    pts, embedding_used, embedding_warn = _project_high_embedding(
        high=(high_sub if high_sub is not None else np.zeros((0, 2), dtype=np.float32)),
        low_fallback=low_sub,
        embedding_source=embedding_source,
        seed=seed,
        auto_tsne_max_points=embedding_auto_tsne_max_points,
        metric=embedding_metric,
        pre_reduce=highdim_pre_reduce,
        pca_components=highdim_pca_components,
        tsne_perplexity=tsne_perplexity,
        tsne_max_iter=tsne_max_iter,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )
    out["embedding_source_requested"] = str(embedding_source)
    out["embedding_source_used"] = str(embedding_used)
    if embedding_warn:
        out["warnings"].extend(embedding_warn)

    payload_path = run_dir / "artifacts" / Path(payload_subpath_clean)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _save_embedding_payload(
        payload_path,
        index=idx,
        projected_points=pts,
        head_points=low_sub,
        high_points=high_sub,
        preds=pred_sub,
        targets=tgt_sub,
        regions=reg_sub,
        uncertainty=unc_sub,
        collision_mask=col_sub,
        embedding_source_used=embedding_used,
    )
    out["embedding_payload_path"] = str(payload_path)
    out["artifact_subdir"] = artifact_subdir_clean

    def _save(fig, name: str) -> None:
        path = vis_dir / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        out["files"].append(str(path))
        try:
            plt.close(fig)
        except Exception:
            pass

    # 1) Embedding by predicted class.
    fig, ax = plt.subplots(figsize=(8, 6))
    if class_groups:
        selected_pred = sorted(configured_group_member_ids)
    else:
        selected_pred = _select_from_source(pred_sub.astype(np.int64, copy=False), strict_group=False)
    pred_alpha_overrides: Dict[int, float] = {}
    pred_counts = Counter([int(x) for x in pred_sub.tolist()])
    pred_total = int(sum(pred_counts.values()))
    out["embedding_pred_dominant_alpha_applied"] = 0
    out["embedding_pred_dominant_class_id"] = -1
    out["embedding_pred_dominant_class_ratio"] = 0.0
    if pred_total > 0 and pred_counts:
        dom_cid, dom_count = max(pred_counts.items(), key=lambda kv: kv[1])
        dom_ratio = float(dom_count) / float(pred_total)
        out["embedding_pred_dominant_class_id"] = int(dom_cid)
        out["embedding_pred_dominant_class_ratio"] = float(dom_ratio)
        if dom_ratio >= float(embedding_pred_dominant_ratio_threshold):
            pred_alpha_overrides[int(dom_cid)] = float(embedding_pred_dominant_alpha)
            out["embedding_pred_dominant_alpha_applied"] = 1
    _plot_embedding_by_labels(
        ax=ax,
        pts=pts,
        labels=pred_sub.astype(np.int64, copy=False),
        class_map=display_class_map,
        class_colors=soft_class_colors,
        selected_ids=selected_pred,
        title="Embedding by Predicted Class",
        legend_max_items=legend_limit,
        class_to_group=class_to_group,
        configured_group_count=len(configured_group_rows),
        show_group_prefix=bool(class_groups),
        show_missing_in_legend=bool(class_groups) and bool(show_zero_count_classes_in_legend),
        scatter_alpha=float(embedding_scatter_alpha),
        alpha_overrides=pred_alpha_overrides,
    )
    _save(fig, "embedding_pred.png")

    # 2) Embedding by target class.
    fig, ax = plt.subplots(figsize=(8, 6))
    selected_target = _select_from_source(tgt_sub.astype(np.int64, copy=False), strict_group=bool(class_groups))
    _plot_embedding_by_labels(
        ax=ax,
        pts=pts,
        labels=tgt_sub.astype(np.int64, copy=False),
        class_map=display_class_map,
        class_colors=soft_class_colors,
        selected_ids=selected_target,
        title="Embedding by Target Class",
        legend_max_items=legend_limit,
        class_to_group=class_to_group,
        configured_group_count=len(configured_group_rows),
        show_group_prefix=bool(class_groups),
        show_missing_in_legend=bool(class_groups) and bool(show_zero_count_classes_in_legend),
        scatter_alpha=float(embedding_scatter_alpha),
    )
    _save(fig, "embedding_target.png")

    # 2.1) Class comparison view (class/topk configurable).
    cmp_src = tgt_sub if compare_source == "target" else pred_sub
    cmp_selected = _select_from_source(cmp_src.astype(np.int64, copy=False), strict_group=bool(class_groups))
    cmp_ordered = _ordered_classes_by_count(cmp_src.astype(np.int64, copy=False), selected_ids=cmp_selected)
    if compare_mode == "topk":
        cmp_plot_ids = cmp_ordered[: min(compare_topk, len(cmp_ordered))]
    else:
        cmp_plot_ids = cmp_ordered
    out["class_compare_mode"] = str(compare_mode)
    out["class_compare_source"] = str(compare_source)
    out["class_compare_topk"] = int(compare_topk)
    out["class_compare_count"] = int(len(cmp_plot_ids))
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_embedding_by_labels(
        ax=ax,
        pts=pts,
        labels=cmp_src.astype(np.int64, copy=False),
        class_map=display_class_map,
        class_colors=soft_class_colors,
        selected_ids=cmp_plot_ids,
        title=(
            f"Embedding by Class Comparison ({compare_source}, top-{compare_topk})"
            if compare_mode == "topk"
            else f"Embedding by Class Comparison ({compare_source}, all classes)"
        ),
        legend_max_items=legend_limit,
        class_to_group=class_to_group,
        configured_group_count=len(configured_group_rows),
        show_group_prefix=bool(class_groups and compare_source == "target"),
        show_missing_in_legend=bool(class_groups) and bool(show_zero_count_classes_in_legend),
        scatter_alpha=float(embedding_scatter_alpha),
    )
    _save(fig, "embedding_class_compare.png")

    # 2.2) Group-aware class view (old grouped class style).
    if class_groups or include_ids:
        src = tgt_sub if focus_label_source == "target" else pred_sub
        fig, ax = plt.subplots(figsize=(9, 7))
        selected = set(_select_from_source(src.astype(np.int64, copy=False), strict_group=bool(class_groups)))
        counts = Counter([int(x) for x in src.tolist()])
        configured_group_count = len(configured_group_rows)
        for cid in sorted(selected, key=lambda c: counts.get(int(c), 0), reverse=True):
            m = src == int(cid)
            if not np.any(m):
                continue
            color = soft_class_colors.get(int(cid), _lighten_rgba(plt.get_cmap("tab20")(cid % 20 / 19.0), float(embedding_color_lighten)))
            gidx = class_to_group.get(int(cid), -1)
            cname = display_class_map.get(int(cid), f"class_{int(cid)}")
            if gidx >= 0 and gidx < configured_group_count:
                label = f"G{gidx + 1}:{cname} ({counts[int(cid)]})"
            else:
                label = f"{cname} ({counts[int(cid)]})"
            ax.scatter(
                pts[m, 0],
                pts[m, 1],
                s=12,
                alpha=float(embedding_scatter_alpha),
                color=color,
                linewidths=0,
                label=label,
            )
        ax.set_title(f"Embedding by Class Groups ({focus_label_source})")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[:legend_limit], l[:legend_limit], loc="best", fontsize=7, frameon=True)
        _save(fig, "embedding_grouped_classes.png")

    # 2.3) Datamap-style semantic landscape figures. These are additive outputs and
    # should be paired with region/collision figures rather than replacing them.
    if datamap_enabled:
        datamap_files: list[str] = []
        datamap_warnings: list[str] = []

        target_labels_dm = _build_datamap_labels(
            tgt_sub.astype(np.int64, copy=False),
            class_map=display_class_map,
            class_to_group=class_to_group,
            configured_group_rows=configured_group_rows,
            mode=("class_with_group" if class_groups else "class"),
        )
        ok, warn = _export_datamapplot_figure(
            points=pts,
            labels=target_labels_dm,
            out_path=vis_dir / "datamap_target.png",
            title="Semantic Landscape by Target Class",
            point_size=datamap_point_size,
            alpha=datamap_alpha,
            min_font_size=datamap_min_font_size,
             max_font_size=datamap_max_font_size,
             label_wrap_width=datamap_label_wrap_width,
             title_font_size=datamap_title_font_size,
             force_matplotlib=datamap_force_matplotlib,
             use_medoids=datamap_use_medoids,
         )
        if ok:
            datamap_files.append(str(vis_dir / "datamap_target.png"))
        elif warn:
            datamap_warnings.append(warn)

        if class_groups:
            group_labels_dm = _build_datamap_labels(
                tgt_sub.astype(np.int64, copy=False),
                class_map=display_class_map,
                class_to_group=class_to_group,
                configured_group_rows=configured_group_rows,
                mode="group",
            )
            ok, warn = _export_datamapplot_figure(
                points=pts,
                labels=group_labels_dm,
                out_path=vis_dir / "datamap_grouped_target.png",
                title="Semantic Landscape by Target Groups",
                point_size=datamap_point_size,
                alpha=datamap_alpha,
                min_font_size=datamap_min_font_size,
                 max_font_size=datamap_max_font_size,
                 label_wrap_width=datamap_label_wrap_width,
                 title_font_size=datamap_title_font_size,
                 force_matplotlib=datamap_force_matplotlib,
                 use_medoids=datamap_use_medoids,
             )
            if ok:
                datamap_files.append(str(vis_dir / "datamap_grouped_target.png"))
            elif warn:
                datamap_warnings.append(warn)

        pred_labels_dm = _build_datamap_labels(
            pred_sub.astype(np.int64, copy=False),
            class_map=display_class_map,
            class_to_group=class_to_group,
            configured_group_rows=configured_group_rows,
            mode=("class_with_group" if class_groups else "class"),
        )
        ok, warn = _export_datamapplot_figure(
            points=pts,
            labels=pred_labels_dm,
            out_path=vis_dir / "datamap_pred.png",
            title="Semantic Landscape by Predicted Class",
            point_size=datamap_point_size,
            alpha=datamap_alpha,
            min_font_size=datamap_min_font_size,
             max_font_size=datamap_max_font_size,
             label_wrap_width=datamap_label_wrap_width,
             title_font_size=datamap_title_font_size,
             force_matplotlib=datamap_force_matplotlib,
             use_medoids=datamap_use_medoids,
         )
        if ok:
            datamap_files.append(str(vis_dir / "datamap_pred.png"))
        elif warn:
            datamap_warnings.append(warn)

        out["files"].extend(datamap_files)
        if datamap_warnings:
            out["warnings"].extend(sorted(set(datamap_warnings)))
        out["datamap_files_count"] = int(len(datamap_files))
    else:
        out["datamap_files_count"] = 0

    # 3) Three-way regions.
    region_colors = {0: "#d73027", 1: "#fdae61", 2: "#1a9850"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for rv in sorted(np.unique(reg_sub).tolist()):
        mask = reg_sub == rv
        if not np.any(mask):
            continue
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=12,
            alpha=0.85,
            c=region_colors.get(int(rv), "#636363"),
            label=_label_name(int(rv)),
            linewidths=0,
        )
    ax.set_title("Three-Way Regions")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend(loc="best", fontsize=8)
    _save(fig, "three_way_regions.png")

    # 4) Uncertainty heat map.
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=unc_sub, s=10, cmap="viridis", alpha=0.9, linewidths=0)
    ax.set_title("Embedding Uncertainty")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Uncertainty")
    _save(fig, "uncertainty_heatmap.png")

    # 5) Collision overlay.
    if col_sub is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        base = ~col_sub.astype(bool)
        if np.any(base):
            ax.scatter(pts[base, 0], pts[base, 1], s=8, c="#bdbdbd", alpha=0.4, label="Non-collision", linewidths=0)
        if np.any(col_sub):
            ax.scatter(
                pts[col_sub, 0],
                pts[col_sub, 1],
                s=18,
                c="#e31a1c",
                alpha=0.9,
                label="Collision",
                linewidths=0,
            )
        ax.set_title("Collision Samples")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.legend(loc="best", fontsize=8)
        _save(fig, "collision_overlay.png")

    # 6) Granular balls projected from member points in 2D with class-aware colors.
    if balls:
        fig, ax = plt.subplots(figsize=(9, 7))
        src = tgt_sub if focus_label_source == "target" else pred_sub
        selected = set(_select_from_source(src.astype(np.int64, copy=False), strict_group=bool(class_groups)))
        if class_groups:
            overlay_focus_ids = sorted(configured_group_member_ids)
        elif include_ids:
            overlay_focus_ids = sorted(include_ids)
        else:
            overlay_focus_ids = []
        global_to_local = {int(g): int(i) for i, g in enumerate(idx.tolist())}

        def _resolve_ball_visual_class(members_global: np.ndarray, fallback_dom: int) -> int:
            if ball_label_mode == "dominant":
                return int(fallback_dom)
            arr = tgt_all if ball_label_mode == "target" else pred_all
            valid = members_global[(members_global >= 0) & (members_global < n_all)]
            if valid.size == 0:
                return int(fallback_dom)
            vals = arr[valid].astype(np.int64, copy=False)
            if vals.size == 0:
                return int(fallback_dom)
            uniq, cnt = np.unique(vals, return_counts=True)
            return int(uniq[int(np.argmax(cnt))])

        for cid in sorted(selected):
            m = src == cid
            if not np.any(m):
                continue
            color = soft_class_colors.get(int(cid), _lighten_rgba(plt.get_cmap("tab20")(cid % 20 / 19.0), float(embedding_color_lighten)))
            ax.scatter(pts[m, 0], pts[m, 1], s=8, c=[color], alpha=0.35, linewidths=0)

        overlay_count = 0
        dom_counter: Counter[int] = Counter()
        ball_centers_2d: list[tuple[float, float]] = []
        ball_radii_2d: list[float] = []
        ball_dom_list: list[int] = []
        for b in balls:
            if overlay_count >= max_ball_overlays:
                break
            members = np.asarray(getattr(b, "members", []), dtype=np.int64)
            if members.size < 2:
                continue
            local_members = [global_to_local[int(m)] for m in members.tolist() if int(m) in global_to_local]
            if len(local_members) < 2:
                continue
            local = pts[np.asarray(local_members, dtype=np.int64)]
            c2 = np.mean(local, axis=0)
            r2 = float(np.max(np.linalg.norm(local - c2[None, :], axis=1)))
            if not np.isfinite(r2) or r2 <= 0.0:
                continue
            dom = int(getattr(b, "dominant_class", -1))
            vis_cls = _resolve_ball_visual_class(members.astype(np.int64, copy=False), dom)
            dom_counter[vis_cls] += 1
            color = class_colors.get(vis_cls, plt.get_cmap("tab20")(vis_cls % 20 / 19.0))
            circ = Circle((float(c2[0]), float(c2[1])), r2, fill=False, linewidth=1.1, edgecolor=color, alpha=0.6)
            ax.add_patch(circ)
            ball_centers_2d.append((float(c2[0]), float(c2[1])))
            ball_radii_2d.append(float(r2))
            ball_dom_list.append(vis_cls)
            overlay_count += 1

        legend_handles = []
        if overlay_focus_ids:
            for cid in overlay_focus_ids:
                cnt = int(dom_counter.get(int(cid), 0))
                if cnt <= 0 and not bool(show_zero_count_classes_in_legend):
                    continue
                cname = display_class_map.get(int(cid), f"class_{int(cid)}")
                color = class_colors.get(int(cid), plt.get_cmap("tab20")(cid % 20 / 19.0))
                pref = _group_prefix(int(cid), class_to_group, configured_group_rows) if class_groups else ""
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=(2.0 if cnt > 0 else 1.0),
                        alpha=(0.95 if cnt > 0 else 0.45),
                        linestyle=("-" if cnt > 0 else "--"),
                        label=f"{pref}{cname} balls={cnt}",
                    )
                )
        else:
            for dom, cnt in dom_counter.most_common(10):
                cname = display_class_map.get(int(dom), f"class_{int(dom)}")
                color = class_colors.get(int(dom), plt.get_cmap("tab20")(dom % 20 / 19.0))
                legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=f"{cname} balls={cnt}"))
        if legend_handles:
            ax.legend(handles=legend_handles[:legend_limit], loc="best", fontsize=8, frameon=True)

        ax.set_title(f"Projected Granular Balls by Ball Label ({ball_label_mode})")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        out["ball_overlays"] = int(overlay_count)
        out["ball_unique_classes_plotted"] = int(len(dom_counter))
        _save(fig, "granular_balls_overlay.png")

        # 7) 3D granular balls view: (x, y, projected_radius), radius-bucket coloring.
        if balls_3d_enabled and ball_radii_2d:
            try:
                fig = plt.figure(figsize=(10, 7))
                ax3 = fig.add_subplot(111, projection="3d")
                rr = np.asarray(ball_radii_2d, dtype=np.float64)
                q1 = float(np.quantile(rr, float(qvals[0])))
                q2 = float(np.quantile(rr, float(qvals[1])))
                out["balls_3d_radius_thresholds"] = [q1, q2]
                bucket_colors = {
                    "small": "#2ca25f",
                    "medium": "#fdae6b",
                    "large": "#de2d26",
                }
                bucket_markers = {
                    "small": "o",
                    "medium": "^",
                    "large": "s",
                }
                bucket_counts: Counter[str] = Counter()
                class_counts: Counter[int] = Counter()
                for (cx, cy), r2, dom in zip(ball_centers_2d, ball_radii_2d, ball_dom_list):
                    if r2 <= q1:
                        bucket = "small"
                    elif r2 <= q2:
                        bucket = "medium"
                    else:
                        bucket = "large"
                    bucket_counts[bucket] += 1
                    class_counts[int(dom)] += 1
                    edge = class_colors.get(int(dom), (0.2, 0.2, 0.2, 1.0))
                    ax3.scatter(
                        [cx],
                        [cy],
                        [r2],
                        s=58,
                        c=[bucket_colors[bucket]],
                        marker=bucket_markers[bucket],
                        alpha=0.92,
                        edgecolors=[edge],
                        linewidths=0.8,
                    )

                from matplotlib.patches import Patch

                radius_handles = [
                    Patch(facecolor=bucket_colors["small"], label=f"small (<=Q{int(qvals[0]*100)})"),
                    Patch(facecolor=bucket_colors["medium"], label=f"medium (Q{int(qvals[0]*100)}-Q{int(qvals[1]*100)})"),
                    Patch(facecolor=bucket_colors["large"], label=f"large (>Q{int(qvals[1]*100)})"),
                ]
                class_handles = []
                if overlay_focus_ids:
                    for cid in overlay_focus_ids:
                        cnt = int(class_counts.get(int(cid), 0))
                        if cnt <= 0 and not bool(show_zero_count_classes_in_legend):
                            continue
                        cname = display_class_map.get(int(cid), f"class_{int(cid)}")
                        c = class_colors.get(int(cid), (0.2, 0.2, 0.2, 1.0))
                        pref = _group_prefix(int(cid), class_to_group, configured_group_rows) if class_groups else ""
                        class_handles.append(
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                linestyle="",
                                markerfacecolor="white",
                                markeredgecolor=c,
                                markeredgewidth=(2.0 if cnt > 0 else 1.0),
                                alpha=(0.95 if cnt > 0 else 0.45),
                                label=f"{pref}{cname} balls={cnt}",
                            )
                        )
                else:
                    for dom, cnt in class_counts.most_common(8):
                        cname = display_class_map.get(int(dom), f"class_{int(dom)}")
                        c = class_colors.get(int(dom), (0.2, 0.2, 0.2, 1.0))
                        class_handles.append(
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                linestyle="",
                                markerfacecolor="white",
                                markeredgecolor=c,
                                markeredgewidth=2.0,
                                label=f"{cname} balls={cnt}",
                            )
                        )
                ax3.legend(
                    handles=radius_handles + class_handles[: max(0, legend_limit - len(radius_handles))],
                    loc="best",
                    fontsize=8,
                    frameon=True,
                )
                ax3.set_title(f"3D Granular Balls (label={ball_label_mode}, Z=Projected Radius, Color=Radius Bucket)")
                ax3.set_xlabel("Dim-1")
                ax3.set_ylabel("Dim-2")
                ax3.set_zlabel("Projected Radius")
                out["balls_3d_count"] = int(len(ball_radii_2d))
                _save(fig, "granular_balls_3d.png")
            except Exception:
                out["warnings"].append("balls_3d_plot_failed")

    return out


def export_embedding_projection_comparison(
    *,
    run_dir: Path,
    embedding_compare_sources: Sequence[Any] | None,
    export_kwargs: Mapping[str, Any],
    comparison_root_subdir: str = "visualization_ablation",
) -> Dict[str, Any]:
    comparison_root_clean = str(comparison_root_subdir).strip().replace("\\", "/").strip("/") or "visualization_ablation"
    out: Dict[str, Any] = {
        "enabled": False,
        "comparison_root_subdir": comparison_root_clean,
        "requested_sources": [],
        "rows": [],
        "warnings": [],
        "summary_json_path": "",
        "summary_csv_path": "",
    }
    if not embedding_compare_sources:
        return out

    requested_sources: list[str] = []
    seen: set[str] = set()
    for token in embedding_compare_sources:
        source = str(token).strip()
        if not source or source in seen:
            continue
        seen.add(source)
        if source not in _VALID_EMBEDDING_SOURCES:
            out["warnings"].append(f"invalid_compare_source:{source}")
            continue
        requested_sources.append(source)
    if not requested_sources:
        return out

    out["enabled"] = True
    out["requested_sources"] = list(requested_sources)
    comparison_root = run_dir / "artifacts" / Path(comparison_root_clean)
    comparison_root.mkdir(parents=True, exist_ok=True)
    default_source = str(export_kwargs.get("embedding_source", "head"))

    rows: list[Dict[str, Any]] = []
    warnings_out: list[str] = list(out["warnings"])
    for source in requested_sources:
        tag = source.replace("/", "_").replace("\\", "_").replace(":", "_")
        artifact_subdir = f"{comparison_root_clean}/{tag}"
        call_kwargs = dict(export_kwargs)
        call_kwargs["artifact_subdir"] = artifact_subdir
        call_kwargs["embedding_payload_subpath"] = f"{artifact_subdir}/embedding_payload.npz"
        call_kwargs["embedding_source"] = source
        meta = export_visualization_artifacts(run_dir=run_dir, **call_kwargs)
        row: Dict[str, Any] = {
            "embedding_source_requested": source,
            "embedding_source_used": str(meta.get("embedding_source_used", "")),
            "is_default_embedding_source": int(source == default_source),
            "artifact_subdir": str(meta.get("artifact_subdir", artifact_subdir)),
            "embedding_payload_path": str(meta.get("embedding_payload_path", "")),
            "num_points_plotted": int(meta.get("num_points_plotted", 0)),
            "datamap_files_count": int(meta.get("datamap_files_count", 0)),
            "warnings": list(meta.get("warnings", [])),
        }
        rows.append(row)
        meta_path = run_dir / "artifacts" / Path(artifact_subdir) / "visualization_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        row["visualization_meta_path"] = str(meta_path)
        warnings_out.extend([f"{source}:{w}" for w in row["warnings"]])

    summary_json_path = comparison_root / "summary.json"
    summary_csv_path = comparison_root / "summary.csv"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_sources": requested_sources,
                "rows": rows,
                "warnings": sorted(set(warnings_out)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "embedding_source_requested",
                "embedding_source_used",
                "is_default_embedding_source",
                "artifact_subdir",
                "embedding_payload_path",
                "visualization_meta_path",
                "num_points_plotted",
                "datamap_files_count",
                "warnings",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "warnings": ";".join(str(x) for x in row.get("warnings", [])),
                }
            )

    out["rows"] = rows
    out["warnings"] = sorted(set(warnings_out))
    out["summary_json_path"] = str(summary_json_path)
    out["summary_csv_path"] = str(summary_csv_path)
    return out
