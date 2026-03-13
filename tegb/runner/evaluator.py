from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict

import torch

from tegb.config.schema import ExperimentConfig
from tegb.runner.trainer import ResearchTrainer


class ResearchEvaluator:
    def __init__(self, cfg: ExperimentConfig, ckpt_path: str) -> None:
        # Eval path always uses full configured dataset without train/val re-splitting.
        self.cfg = copy.deepcopy(cfg)
        self.cfg.data.val_split_ratio = 0.0
        self.ckpt_path = Path(ckpt_path)
        self.trainer = ResearchTrainer(self.cfg)
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.trainer.evidential_head.load_state_dict(ckpt["evidential_head"])
        self.trainer.proj_head.load_state_dict(ckpt["proj_head"])

    def run(self) -> Dict[str, float]:
        try:
            metrics = self.trainer.evaluate(stream_to_disk=True)
            out_path = self.trainer.run_dir / "eval_metrics.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            return metrics
        finally:
            self.trainer.close()
