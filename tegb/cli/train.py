from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tegb.config import load_config
from tegb.config.schema import ExperimentConfig
from tegb.runner import ResearchTrainer


def _resolve_unique_exp_name(cfg: ExperimentConfig) -> str:
    base = str(cfg.experiment.exp_name).strip() or "tegb_experiment"
    root = Path(cfg.experiment.output_root)
    candidate = base
    idx = 1
    while (root / candidate).exists():
        candidate = f"{base}_{idx}"
        idx += 1
    return candidate


def main(config_path: str, exp_name: str | None = None) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    if exp_name is not None and str(exp_name).strip():
        cfg.experiment.exp_name = str(exp_name).strip()

    final_name = _resolve_unique_exp_name(cfg)
    if final_name != cfg.experiment.exp_name:
        print(f"[TEGB] exp_name already exists, auto-renamed to: {final_name}")
    cfg.experiment.exp_name = final_name

    trainer = ResearchTrainer(cfg)
    metrics = trainer.fit()
    return metrics


def _entry() -> None:
    parser = argparse.ArgumentParser(description="TEGB-YOLO train entrypoint")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--exp-name", default=None, help="Override experiment name; default uses YAML exp_name")
    args = parser.parse_args()
    metrics = main(args.config, exp_name=args.exp_name)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _entry()
