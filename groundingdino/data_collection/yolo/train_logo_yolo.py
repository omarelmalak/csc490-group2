"""Train YOLOv8n class-agnostic logo detector (Untitled10_organized §6)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from config import (
    LOGO_EPOCHS,
    LOGO_WEIGHTS,
    MERGED_LOGO_DIR,
    YOLO_IMGSZ,
    YOLO_RUNS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train class-agnostic logo YOLO detector.")
    parser.add_argument("--epochs", type=int, default=LOGO_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()

    data_yaml = MERGED_LOGO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing {data_yaml}. Run download_datasets.py then build_merged_logo_dataset.py."
        )

    print("Training logo detector on", data_yaml)
    print("Device:", args.device)

    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(YOLO_RUNS_DIR),
        name="logo_detector",
        exist_ok=True,
        verbose=True,
    )

    best = YOLO_RUNS_DIR / "logo_detector" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected weights missing: {best}")

    dest_dir = LOGO_WEIGHTS.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, LOGO_WEIGHTS)
    print(f"Saved logo weights -> {LOGO_WEIGHTS}")


if __name__ == "__main__":
    main()
