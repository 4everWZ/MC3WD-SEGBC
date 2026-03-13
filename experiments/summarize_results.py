from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

PRIMARY_METRICS = ["vsf_auc", "gtf_auc", "snh_auc", "fpr95", "coverage_at_risk"]
LOWER_IS_BETTER = {
    "fpr95",
    "davies_high",
    "davies_2d",
    "warning_cov_fallback",
    "warning_chi2_fallback",
    "warning_split_fallback",
    "warning_split_non_converged",
    "warning_ph_fallback",
    "builder_exception",
    "v3_uncovered_rate",
}


def _try_float(v: object) -> float | None:
    if isinstance(v, (int, float)):
        fv = float(v)
        return fv if np.isfinite(fv) else None
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            fv = float(s)
            return fv if np.isfinite(fv) else None
        except Exception:
            return None
    return None


def _is_higher_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def _first_available_pvalue(row: Dict[str, object]) -> float | None:
    for key in ("p_permutation", "p_welch", "p_mannwhitney"):
        v = _try_float(row.get(key))
        if v is not None:
            return float(v)
    return None


def _mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _sem(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1) / np.sqrt(len(vals)))


def _ci95(vals: List[float]) -> float:
    return 1.96 * _sem(vals)


def _welch_pvalue(a: List[float], b: List[float]) -> float | None:
    try:
        from scipy.stats import ttest_ind  # type: ignore

        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def _mannwhitney_pvalue(a: List[float], b: List[float]) -> float | None:
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return None


def _permutation_pvalue(a: List[float], b: List[float], permutations: int = 5000, seed: int = 42) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    obs = abs(float(np.mean(a_arr) - np.mean(b_arr)))
    pool = np.concatenate([a_arr, b_arr], axis=0)
    n_a = a_arr.shape[0]
    rng = np.random.default_rng(seed)

    more_extreme = 0
    for _ in range(max(100, int(permutations))):
        rng.shuffle(pool)
        diff = abs(float(np.mean(pool[:n_a]) - np.mean(pool[n_a:])))
        if diff >= obs:
            more_extreme += 1
    return float((more_extreme + 1) / (max(100, int(permutations)) + 1))


def _cohens_d(a: List[float], b: List[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape[0] < 2 or b_arr.shape[0] < 2:
        return 0.0
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    pooled = ((a_arr.shape[0] - 1) * var_a + (b_arr.shape[0] - 1) * var_b) / (a_arr.shape[0] + b_arr.shape[0] - 2)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / np.sqrt(pooled))


def _load_csv_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, object] = {}
            for k, v in r.items():
                if v is None:
                    continue
                fv = _try_float(v)
                row[k] = fv if fv is not None else v
            rows.append(row)
    return rows


def _collect_ablation_rows(root: Path, explicit_ablation_csv: str | None = None) -> List[Dict[str, object]]:
    tables: List[Path] = []
    if explicit_ablation_csv:
        tables = [Path(explicit_ablation_csv)]
    else:
        tables = sorted(root.rglob("ablation_table.csv"))

    rows: List[Dict[str, object]] = []
    for t in tables:
        if not t.exists():
            continue
        for r in _load_csv_rows(t):
            r["table_path"] = str(t)
            if "run_dir" not in r:
                r["run_dir"] = str(t.parent)
            rows.append(r)
    return rows


def _collect_history_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in root.rglob("metrics.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            hist = obj.get("history") or []
            if not hist:
                continue
            last = hist[-1]
            row: Dict[str, object] = {"run_dir": str(p.parent), "variant": "single", "seed": -1}
            row.update(last)
            rows.append(row)
        except Exception:
            continue
    return rows


def _metric_keys(rows: List[Dict[str, object]]) -> List[str]:
    ignore = {"variant", "seed", "run_dir", "table_path", "epoch"}
    keys = set()
    for r in rows:
        for k, v in r.items():
            if k in ignore:
                continue
            if isinstance(v, (int, float)):
                keys.add(k)
    return sorted(keys)


def _summary_long(rows: List[Dict[str, object]], metrics: List[str]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        variant = str(r.get("variant", "unknown"))
        for m in metrics:
            fv = _try_float(r.get(m))
            if fv is not None:
                grouped[variant][m].append(fv)

    out: List[Dict[str, object]] = []
    for variant, md in grouped.items():
        for metric in metrics:
            vals = md.get(metric, [])
            if not vals:
                continue
            mean_v, std_v = _mean_std(vals)
            out.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": mean_v,
                    "std": std_v,
                    "sem": _sem(vals),
                    "ci95": _ci95(vals),
                    "n": len(vals),
                }
            )
    return out


def _summary_wide(summary_long: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_variant: Dict[str, Dict[str, object]] = defaultdict(dict)
    metrics = sorted({str(r["metric"]) for r in summary_long})
    for r in summary_long:
        v = str(r["variant"])
        m = str(r["metric"])
        by_variant[v]["variant"] = v
        by_variant[v][f"{m}_mean"] = float(r["mean"])
        by_variant[v][f"{m}_std"] = float(r["std"])
        by_variant[v][f"{m}_n"] = int(r["n"])

    rows = [by_variant[v] for v in sorted(by_variant.keys(), key=lambda x: (x != "Full", x))]
    for row in rows:
        for m in metrics:
            row.setdefault(f"{m}_mean", "")
            row.setdefault(f"{m}_std", "")
            row.setdefault(f"{m}_n", "")
    return rows


def _significance(
    rows: List[Dict[str, object]],
    metrics: List[str],
    full_variant: str = "Full",
    permutations: int = 5000,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        variant = str(r.get("variant", "unknown"))
        for m in metrics:
            fv = _try_float(r.get(m))
            if fv is not None:
                grouped[variant][m].append(fv)

    if full_variant not in grouped:
        return []

    out: List[Dict[str, object]] = []
    for metric in metrics:
        full_vals = grouped[full_variant].get(metric, [])
        if len(full_vals) < 2:
            continue
        full_mean = float(np.mean(full_vals))
        for variant, md in grouped.items():
            if variant == full_variant:
                continue
            vals = md.get(metric, [])
            if len(vals) < 2:
                continue
            var_mean = float(np.mean(vals))
            raw_delta = full_mean - var_mean
            objective_delta = raw_delta if _is_higher_better(metric) else -raw_delta
            out.append(
                {
                    "metric": metric,
                    "baseline_variant": full_variant,
                    "compare_variant": variant,
                    "baseline_mean": full_mean,
                    "compare_mean": var_mean,
                    "delta_baseline_minus_compare": raw_delta,
                    "objective_delta_positive_is_baseline_better": objective_delta,
                    "effect_direction": "baseline_better" if objective_delta > 0 else "baseline_worse",
                    "cohens_d": _cohens_d(full_vals, vals),
                    "p_welch": _welch_pvalue(full_vals, vals),
                    "p_mannwhitney": _mannwhitney_pvalue(full_vals, vals),
                    "p_permutation": _permutation_pvalue(full_vals, vals, permutations=permutations),
                    "n_baseline": len(full_vals),
                    "n_compare": len(vals),
                }
            )
    return out


def _auto_conclusions(
    significance_rows: List[Dict[str, object]],
    required_metrics: Iterable[str],
    p_threshold: float,
    min_support_ratio: float,
    baseline_variant: str = "Full",
) -> Dict[str, object]:
    required = list(required_metrics)
    by_variant_metric: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in significance_rows:
        variant = str(row.get("compare_variant", ""))
        metric = str(row.get("metric", ""))
        if variant and metric:
            by_variant_metric[variant][metric] = row

    variants: List[Dict[str, object]] = []
    compared_variants = sorted(by_variant_metric.keys())
    for variant in compared_variants:
        metric_details: List[Dict[str, object]] = []
        support_count = 0
        available_count = 0
        for metric in required:
            r = by_variant_metric[variant].get(metric)
            if r is None:
                metric_details.append(
                    {
                        "metric": metric,
                        "available": False,
                        "supported": False,
                        "reason": "missing",
                    }
                )
                continue
            available_count += 1
            p_value = _first_available_pvalue(r)
            significant = (p_value is not None) and (p_value < p_threshold)
            direction_ok = _try_float(r.get("objective_delta_positive_is_baseline_better")) is not None and float(
                r["objective_delta_positive_is_baseline_better"]
            ) > 0.0
            supported = bool(significant and direction_ok)
            if supported:
                support_count += 1
            reason = "ok"
            if not significant:
                reason = "not_significant"
            elif not direction_ok:
                reason = "wrong_direction"
            metric_details.append(
                {
                    "metric": metric,
                    "available": True,
                    "supported": supported,
                    "p_value_selected": p_value,
                    "effect_direction": r.get("effect_direction"),
                    "reason": reason,
                }
            )

        expected = len(required)
        support_ratio_expected = float(support_count / expected) if expected > 0 else 0.0
        support_ratio_available = float(support_count / available_count) if available_count > 0 else 0.0
        coverage_ratio = float(available_count / expected) if expected > 0 else 0.0
        strict_pass = coverage_ratio == 1.0 and support_ratio_expected >= min_support_ratio
        status = "pass" if strict_pass else ("insufficient_metrics" if coverage_ratio < 1.0 else "fail")
        variants.append(
            {
                "baseline_variant": baseline_variant,
                "compare_variant": variant,
                "status": status,
                "strict_pass": strict_pass,
                "required_metric_count": expected,
                "available_metric_count": available_count,
                "supported_metric_count": support_count,
                "coverage_ratio": coverage_ratio,
                "support_ratio_expected": support_ratio_expected,
                "support_ratio_available": support_ratio_available,
                "metric_details": metric_details,
            }
        )

    strong_support_count = int(sum(1 for v in variants if bool(v["strict_pass"])))
    total_compares = int(len(variants))
    overall = {
        "baseline_variant": baseline_variant,
        "compared_variant_count": total_compares,
        "strict_support_count": strong_support_count,
        "baseline_claim_supported_against_all": total_compares > 0 and strong_support_count == total_compares,
    }
    return {
        "rule": {
            "p_threshold": float(p_threshold),
            "min_support_ratio": float(min_support_ratio),
            "required_metrics": required,
            "pvalue_priority": ["p_permutation", "p_welch", "p_mannwhitney"],
            "direction_rule": "objective_delta_positive_is_baseline_better > 0",
            "strict_pass_requires_full_metric_coverage": True,
        },
        "overall": overall,
        "variants": variants,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    cols_set = set()
    for r in rows:
        cols_set.update(r.keys())
    cols = sorted(cols_set)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _plot_publication_figs(summary_long: List[Dict[str, object]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    by_metric: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in summary_long:
        by_metric[str(r["metric"])].append(r)

    plotted: List[str] = []
    fig_dir = out_dir / "publication_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_for_fig = [m for m in PRIMARY_METRICS if m in by_metric]
    if not metrics_for_fig:
        metrics_for_fig = sorted(by_metric.keys())[:6]

    for metric in metrics_for_fig:
        rows = sorted(by_metric[metric], key=lambda x: (str(x["variant"]) != "Full", str(x["variant"])))
        labels = [str(r["variant"]) for r in rows]
        means = [float(r["mean"]) for r in rows]
        stds = [float(r["std"]) for r in rows]
        colors = ["#2E86AB" if l == "Full" else "#A9A9A9" for l in labels]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("score")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        png_path = fig_dir / f"{metric}_bar.png"
        svg_path = fig_dir / f"{metric}_bar.svg"
        fig.savefig(png_path, dpi=200)
        fig.savefig(svg_path)
        plt.close(fig)
        plotted.extend([str(png_path), str(svg_path)])

    if len(metrics_for_fig) >= 2:
        cols = 2
        rows_n = int(np.ceil(len(metrics_for_fig) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4 * rows_n))
        axes_arr = np.atleast_1d(axes).reshape(rows_n, cols)
        for idx, metric in enumerate(metrics_for_fig):
            r = idx // cols
            c = idx % cols
            ax = axes_arr[r, c]
            mrows = sorted(by_metric[metric], key=lambda x: (str(x["variant"]) != "Full", str(x["variant"])))
            labels = [str(x["variant"]) for x in mrows]
            means = [float(x["mean"]) for x in mrows]
            stds = [float(x["std"]) for x in mrows]
            x = np.arange(len(labels))
            colors = ["#2E86AB" if l == "Full" else "#A9A9A9" for l in labels]
            ax.bar(x, means, yerr=stds, color=colors, capsize=3)
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        for idx in range(len(metrics_for_fig), rows_n * cols):
            r = idx // cols
            c = idx % cols
            axes_arr[r, c].axis("off")
        fig.tight_layout()
        combo_png = fig_dir / "main_metrics_grid.png"
        combo_svg = fig_dir / "main_metrics_grid.svg"
        fig.savefig(combo_png, dpi=220)
        fig.savefig(combo_svg)
        plt.close(fig)
        plotted.extend([str(combo_png), str(combo_svg)])

    return plotted


def _build_manifest(rows: List[Dict[str, object]], out_dir: Path) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    for r in rows:
        run_dir = str(r.get("run_dir", ""))
        if not run_dir:
            continue
        cfg = Path(run_dir) / "config_snapshot.json"
        items.append(
            {
                "variant": r.get("variant"),
                "seed": r.get("seed"),
                "run_dir": run_dir,
                "config_snapshot": str(cfg) if cfg.exists() else None,
            }
        )
    manifest = {"count": len(items), "items": items}
    _write_json(out_dir / "config_manifest.json", manifest)
    return manifest


def main(
    root: str = "runs",
    out_dir: str | None = None,
    ablation_csv: str | None = None,
    out_csv: str | None = None,
    permutations: int = 5000,
    p_threshold: float = 0.05,
    min_support_ratio: float = 1.0,
) -> str:
    root_path = Path(root)
    analysis_dir = Path(out_dir) if out_dir else root_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_ablation_rows(root_path, explicit_ablation_csv=ablation_csv)
    if not rows:
        rows = _collect_history_rows(root_path)
    if not rows:
        raise RuntimeError(f"No experiment rows found under: {root_path}")

    metrics = _metric_keys(rows)
    summary_long = _summary_long(rows, metrics)
    summary_wide = _summary_wide(summary_long)
    significance = _significance(rows, metrics, full_variant="Full", permutations=permutations)
    conclusion = _auto_conclusions(
        significance_rows=significance,
        required_metrics=[m for m in PRIMARY_METRICS if m in metrics],
        p_threshold=p_threshold,
        min_support_ratio=min_support_ratio,
        baseline_variant="Full",
    )
    figs = _plot_publication_figs(summary_long, analysis_dir)
    manifest = _build_manifest(rows, analysis_dir)

    rows_csv = analysis_dir / "ablation_rows_consolidated.csv"
    summary_long_csv = Path(out_csv) if out_csv else analysis_dir / "ablation_summary_long.csv"
    summary_wide_csv = analysis_dir / "ablation_summary_wide.csv"
    significance_csv = analysis_dir / "significance_tests.csv"

    _write_csv(rows_csv, rows)
    _write_csv(summary_long_csv, summary_long)
    _write_csv(summary_wide_csv, summary_wide)
    _write_csv(significance_csv, significance)

    _write_json(
        analysis_dir / "significance.json",
        {"comparisons": significance, "permutations": int(permutations), "full_variant": "Full"},
    )
    _write_json(analysis_dir / "conclusion_report.json", conclusion)
    _write_json(
        analysis_dir / "analysis_manifest.json",
        {
            "rows_csv": str(rows_csv),
            "summary_long_csv": str(summary_long_csv),
            "summary_wide_csv": str(summary_wide_csv),
            "significance_csv": str(significance_csv),
            "conclusion_report_json": str(analysis_dir / "conclusion_report.json"),
            "figures": figs,
            "metrics": metrics,
            "config_manifest_count": manifest["count"],
            "significance_gate": {
                "p_threshold": float(p_threshold),
                "min_support_ratio": float(min_support_ratio),
            },
            "baseline_claim_supported_against_all": conclusion["overall"]["baseline_claim_supported_against_all"],
        },
    )
    return str(summary_long_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize TEGB experiment results (research-grade)")
    parser.add_argument("--root", default="runs", help="Runs root directory")
    parser.add_argument("--out-dir", default=None, help="Output analysis directory")
    parser.add_argument("--ablation-csv", default=None, help="Optional explicit ablation_table.csv")
    parser.add_argument("--out-csv", default=None, help="Optional output csv path (summary long)")
    parser.add_argument("--permutations", type=int, default=5000, help="Permutation test iterations")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="Significance threshold for auto conclusions")
    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=1.0,
        help="Required ratio of supported primary metrics for strict pass",
    )
    args = parser.parse_args()
    print(
        main(
            args.root,
            args.out_dir,
            args.ablation_csv,
            args.out_csv,
            args.permutations,
            args.p_threshold,
            args.min_support_ratio,
        )
    )
