from __future__ import annotations

import argparse

from tegb.config import load_config
from tegb.runner import AblationRunner


def main(config_path: str) -> str:
    cfg = load_config(config_path)
    runner = AblationRunner(cfg)
    out = runner.run()
    return str(out)


def _entry() -> None:
    parser = argparse.ArgumentParser(description="TEGB-YOLO ablation entrypoint")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()
    out = main(args.config)
    print(out)


if __name__ == "__main__":
    _entry()

