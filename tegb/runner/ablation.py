from __future__ import annotations

import copy
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List

from tegb.config.schema import ExperimentConfig
from tegb.runner.trainer import ResearchTrainer
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


def _progress(iterable, total: int | None = None, desc: str = "", leave: bool = False):
    if _tqdm is None:
        return iterable
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)() or getattr(sys.stderr, "isatty", lambda: False)())
    return _tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True, disable=not is_tty)


class AblationRunner:
    def __init__(self, base_cfg: ExperimentConfig) -> None:
        self.base_cfg = base_cfg

    def _variant_cfg(self, variant: str, seed: int) -> ExperimentConfig:
        cfg = copy.deepcopy(self.base_cfg)
        cfg.experiment.seed = seed
        cfg.experiment.exp_name = f"{self.base_cfg.experiment.exp_name}_{variant}_seed{seed}"
        cfg.loss.enable_tgb = True
        cfg.loss.enable_edl = True
        cfg.loss.enable_gw = True
        cfg.loss.enable_ph = True
        cfg.loss.enable_3wd = True

        if variant == "-TGB":
            cfg.loss.enable_tgb = False
        elif variant == "-EDL":
            cfg.loss.enable_edl = False
            cfg.loss.lambda_edl = 0.0
        elif variant == "-GW":
            cfg.loss.enable_gw = False
            cfg.loss.lambda_gw = 0.0
        elif variant == "-PH":
            cfg.loss.enable_ph = False
            cfg.loss.lambda_ph = 0.0
        elif variant == "-3WD":
            cfg.loss.enable_3wd = False
        elif variant == "-MAXR":
            cfg.granular.radius_policy = "max"
        return cfg

    def run(
        self,
        *,
        progress_callback: Callable[[], None] | None = None,
        show_progress: bool = True,
        progress_desc: str = "TEGB Ablation Runs",
    ) -> Path:
        rows: List[Dict[str, float | str | int]] = []
        tasks = [(variant, seed) for variant in self.base_cfg.ablation.variants for seed in self.base_cfg.ablation.seeds]
        iterator = _progress(tasks, total=len(tasks), desc=progress_desc, leave=True) if show_progress else tasks
        for variant, seed in iterator:
            cfg = self._variant_cfg(variant, seed)
            trainer = ResearchTrainer(cfg)
            metrics = trainer.fit()
            row: Dict[str, float | str | int] = {
                "variant": variant,
                "seed": seed,
                "run_dir": str(cfg.run_dir),
            }
            row.update(metrics)
            rows.append(row)
            if progress_callback is not None:
                progress_callback()

        out_csv = self.base_cfg.run_dir / "ablation_table.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            keys = list(rows[0].keys())
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

            summary_csv = self.base_cfg.run_dir / "ablation_summary.csv"
            metrics_by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                v = str(r["variant"])
                for k, val in r.items():
                    if k in {"variant", "seed", "run_dir"}:
                        continue
                    if isinstance(val, (int, float)):
                        metrics_by_variant[v][k].append(float(val))
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                cols = ["variant", "metric", "mean", "std", "n"]
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                for v, metrics in metrics_by_variant.items():
                    for k, vals in metrics.items():
                        writer.writerow(
                            {
                                "variant": v,
                                "metric": k,
                                "mean": mean(vals),
                                "std": (stdev(vals) if len(vals) > 1 else 0.0),
                                "n": len(vals),
                            }
                        )
        return out_csv
