from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution: `python experiments/run_eval.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tegb.cli.eval import main


def entry() -> None:
    parser = argparse.ArgumentParser(description="Run TEGB evaluation")
    parser.add_argument("--config", default='configs/tegb/yolo11_coco128.yaml' , help="Config yaml path")
    parser.add_argument("--ckpt", default='runs/tegb_yolo11_coco128/checkpoints/last.pt', help="Checkpoint path")
    parser.add_argument("--class-compare-mode", default=None, choices=["class", "topk"], help="Override visualization.class_compare_mode")
    parser.add_argument("--class-compare-topk", type=int, default=None, help="Override visualization.class_compare_topk")
    parser.add_argument("--class-compare-label-source", default=None, choices=["target", "pred"], help="Override visualization.class_compare_label_source")
    args = parser.parse_args()
    metrics = main(
        args.config,
        args.ckpt,
        class_compare_mode=args.class_compare_mode,
        class_compare_topk=args.class_compare_topk,
        class_compare_label_source=args.class_compare_label_source,
    )
    print(metrics)


if __name__ == "__main__":
    entry()
