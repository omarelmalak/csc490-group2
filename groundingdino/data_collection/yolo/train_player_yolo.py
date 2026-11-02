"""Train YOLOv8n player detector (Untitled10_organized §5)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from config import (
    PLAYER_EPOCHS,
    PLAYER_NAME_HINTS,
    PLAYER_WEIGHTS,
    YOLO_DATA_DIR,
    YOLO_IMGSZ,
    YOLO_RUNS_DIR,
    YOLO_WEIGHTS_DIR,
)


def yolo_class_names(data_yaml: dict) -> list:
    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train player YOLO detector.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=YOLO_DATA_DIR / "roboflow" / "manifest.json",
    )
    parser.add_argument("--epochs", type=int, default=PLAYER_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Missing {args.manifest}. Run download_datasets.py first.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    player_dir = Path(manifest["player"])
    data_yaml_path = player_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml in {player_dir}")

    with open(data_yaml_path, encoding="utf-8") as f:
        player_yaml = yaml.safe_load(f) or {}
    player_names = yolo_class_names(player_yaml)
    player_indices = [
        idx
        for idx, name in enumerate(player_names)
        if str(name).strip().lower() in PLAYER_NAME_HINTS
    ]
    if not player_indices:
        print("No joueur/gardien/player match; training all classes.")
        player_indices = list(range(len(player_names)))

    print("Player classes:", [(i, player_names[i]) for i in player_indices])
    print("Device:", args.device)

    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        classes=player_indices,
        project=str(YOLO_RUNS_DIR),
        name="player_detector",
        exist_ok=True,
        verbose=True,
    )

    best = YOLO_RUNS_DIR / "player_detector" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected weights missing: {best}")

    dest_dir = PLAYER_WEIGHTS.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, PLAYER_WEIGHTS)
    print(f"Saved player weights -> {PLAYER_WEIGHTS}")


if __name__ == "__main__":
    main()
