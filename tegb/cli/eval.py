from __future__ import annotations

import argparse
import json

from tegb.config import load_config
from tegb.runner import ResearchEvaluator


def main(
    config_path: str,
    ckpt: str,
    class_compare_mode: str | None = None,
    class_compare_topk: int | None = None,
    class_compare_label_source: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    if class_compare_mode is not None:
        cfg.visualization.class_compare_mode = str(class_compare_mode)
    if class_compare_topk is not None:
        cfg.visualization.class_compare_topk = int(class_compare_topk)
    if class_compare_label_source is not None:
        cfg.visualization.class_compare_label_source = str(class_compare_label_source)
    evaluator = ResearchEvaluator(cfg, ckpt_path=ckpt)
    return evaluator.run()


def _entry() -> None:
    parser = argparse.ArgumentParser(description="TEGB-YOLO evaluation entrypoint")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file path")
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _entry()
