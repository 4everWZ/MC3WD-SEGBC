from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Allow direct execution: `python experiments/run_ablation.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tegb.config import load_config
from tegb.runner import AblationRunner
from experiments.summarize_results import main as summarize_main

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


SUITE_CONFIGS = {
    "yolo11": "configs/tegb/ablation_yolo11.yaml",
    "yolo12": "configs/tegb/ablation_yolo12.yaml",
    "yolo26": "configs/tegb/ablation_yolo26.yaml",
}

SUITE_WEIGHTS = {
    "yolo11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    "yolo12": ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
    "yolo26": ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
}
VALID_SCALES = {"n", "s", "m", "l", "x"}


def _parse_csv(text: str | None) -> List[str]:
    if text is None:
        return []
    out: List[str] = []
    for token in str(text).split(","):
        t = token.strip()
        if t:
            out.append(t)
    return out


def _normalize_scales_or_weights(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for token in tokens:
        t = str(token).strip().lower()
        if not t:
            continue
        # Preferred style: scales n/s/m/l/x
        if t in VALID_SCALES:
            out.append(t)
            continue
        # Backward compatibility: full weight names
        if t.endswith(".pt"):
            out.append(t)
            continue
        raise ValueError(f"Invalid --weights token: {token}. Use scales like n,s,m,l,x or full *.pt names.")
    return out


def _pick_suite_weights(suite_name: str, scales_or_weights: List[str]) -> List[str]:
    all_weights = SUITE_WEIGHTS[suite_name]
    out: List[str] = []
    for token in scales_or_weights:
        if token.endswith(".pt"):
            out.append(token)
            continue
        scale = token
        matched = [w for w in all_weights if Path(w).stem.lower().endswith(scale)]
        if not matched:
            raise ValueError(f"Cannot map scale '{scale}' for suite '{suite_name}'.")
        out.append(matched[0])
    return out


def _replace_weight_scale(weight_name: str, scale: str) -> str:
    stem = Path(weight_name).stem
    suffix = Path(weight_name).suffix or ".pt"
    if stem and stem[-1].lower() in VALID_SCALES:
        return f"{stem[:-1]}{scale}{suffix}"
    return f"{stem}{scale}{suffix}"


def _resolve_jobs(config: str | None, suite: str | None, weights_csv: str | None) -> List[Tuple[str, str | None, str | None]]:
    raw_tokens = _parse_csv(weights_csv)
    tokens = _normalize_scales_or_weights(raw_tokens)
    if not tokens:
        tokens = ["s"]

    if suite is not None:
        suite = str(suite).strip().lower()
        suites: List[str]
        if suite == "all":
            suites = ["yolo11", "yolo12", "yolo26"]
        elif suite in SUITE_CONFIGS:
            suites = [suite]
        else:
            raise ValueError(f"Unknown suite: {suite}")

        jobs: List[Tuple[str, str | None, str | None]] = []
        for suite_name in suites:
            cfg_path = SUITE_CONFIGS[suite_name]
            weights = _pick_suite_weights(suite_name, tokens)
            for weight in weights:
                jobs.append((cfg_path, suite_name, weight))
        return jobs

    if not config:
        raise ValueError("Either --config or --suite must be provided.")

    cfg = load_config(str(config))
    mapped_weights: List[str] = []
    for token in tokens:
        if token.endswith(".pt"):
            mapped_weights.append(token)
        else:
            mapped_weights.append(_replace_weight_scale(cfg.experiment.weights, token))
    return [(str(config), None, w) for w in mapped_weights]


def _scale_tag(weight: str) -> str:
    stem = Path(weight).stem
    if stem:
        return stem
    return str(weight)


def _run_with_override(
    config_path: str,
    suite_name: str | None,
    weight_override: str | None,
    *,
    progress_callback=None,
    show_inner_progress: bool = True,
) -> str:
    cfg = load_config(config_path)
    if weight_override:
        cfg.experiment.weights = str(weight_override)
        tag = _scale_tag(weight_override)
        cfg.experiment.exp_name = f"{cfg.experiment.exp_name}_{tag}"
    runner = AblationRunner(cfg)
    if suite_name and weight_override:
        desc = f"Ablation {suite_name}:{Path(weight_override).stem}"
    elif suite_name:
        desc = f"Ablation {suite_name}"
    else:
        desc = "TEGB Ablation Runs"
    out = runner.run(
        progress_callback=progress_callback,
        show_progress=show_inner_progress,
        progress_desc=desc,
    )
    suite_disp = suite_name or "custom"
    print(
        {
            "suite": suite_disp,
            "weight": cfg.experiment.weights,
            "exp_name": cfg.experiment.exp_name,
            "ablation_csv": str(out),
        }
    )
    return str(out)


def entry() -> None:
    parser = argparse.ArgumentParser(description="Run TEGB ablation matrix")
    parser.add_argument("--config", default=None, help="Config yaml path")
    parser.add_argument(
        "--suite",
        default=None,
        choices=["yolo11", "yolo12", "yolo26", "all"],
        help="Run built-in ablation config suite",
    )
    parser.add_argument(
        "--weights",
        default="s",
        help=(
            "Comma-separated scales (n,s,m,l,x), default=s. "
            "Also accepts full .pt names for backward compatibility."
        ),
    )
    parser.add_argument("--summarize", action="store_true", help="Run summarize pipeline after ablation")
    parser.add_argument("--summary-root", default="runs", help="Root directory for summarization scan")
    parser.add_argument("--summary-out-dir", default=None, help="Optional output directory for analysis artifacts")
    args = parser.parse_args()
    jobs = _resolve_jobs(args.config, args.suite, args.weights)
    # Count total ablation runs across all jobs for a global progress bar.
    total_tasks = 0
    for cfg_path, _, _ in jobs:
        cfg = load_config(cfg_path)
        total_tasks += len(cfg.ablation.variants) * len(cfg.ablation.seeds)

    global_bar = _tqdm(total=total_tasks, desc="TEGB Ablation Total", dynamic_ncols=True) if (_tqdm is not None and total_tasks > 0) else None

    outputs: List[str] = []
    try:
        for cfg_path, suite_name, weight_override in jobs:
            out = _run_with_override(
                cfg_path,
                suite_name,
                weight_override,
                progress_callback=(lambda: global_bar.update(1)) if global_bar is not None else None,
                show_inner_progress=(global_bar is None),
            )
            outputs.append(str(out))
            if args.summarize:
                summary_csv = summarize_main(
                    root=args.summary_root,
                    out_dir=args.summary_out_dir,
                    ablation_csv=out,
                )
                print(summary_csv)
    finally:
        if global_bar is not None:
            global_bar.close()
    if len(outputs) > 1:
        print({"ablation_csvs": outputs})


if __name__ == "__main__":
    entry()
