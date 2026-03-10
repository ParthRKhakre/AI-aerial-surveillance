"""Train YOLO on aerial dataset using config YAML."""

from __future__ import annotations

import argparse

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_aerial_yolov8.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
