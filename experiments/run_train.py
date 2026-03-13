from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution: `python experiments/run_train.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tegb.cli.train import main


def entry() -> None:
    parser = argparse.ArgumentParser(description="Run TEGB training")
    parser.add_argument("--config",default="configs/tegb/yolo11_coco128.yaml", help="Config yaml path")
    parser.add_argument("--exp-name", default=None, help="Override experiment name; default uses YAML exp_name")
    args = parser.parse_args()
    metrics = main(args.config, exp_name=args.exp_name)
    print(metrics)


if __name__ == "__main__":
    entry()
