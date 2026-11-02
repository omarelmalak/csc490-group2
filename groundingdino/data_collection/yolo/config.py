"""Config shared by YOLO download / train / label scripts (mirrors Untitled10_organized)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python yolo/foo.py` imports of sibling package modules.
_DC = Path(__file__).resolve().parent.parent
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from paths import (  # noqa: E402
    LOGO_WEIGHTS,
    MERGED_LOGO_DIR,
    PLAYER_WEIGHTS,
    YOLO_DATA_DIR,
    YOLO_RUNS_DIR,
    YOLO_WEIGHTS_DIR,
)

ROBOFLOW_API_KEY = "A1UDb8W87l9RzmQ4CMlU"

PLAYER_WORKSPACE = "school-hnpey"
PLAYER_PROJECT = "soccer-iqjx5"
PLAYER_VERSION = 2

LOGO_DETECTION_SOURCES = [
    ("sponsor_detector", "syed-fathaullah", "sponsor-detector", 2),
    ("logo_detection_e92p3", "data-deo7j", "logo-detection-e92p3", 4),
]

PLAYER_EPOCHS = 25
LOGO_EPOCHS = 5
YOLO_IMGSZ = 640

PLAYER_NAME_HINTS = {"joueur", "gardien", "player", "goalkeeper", "keeper"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

__all__ = [
    "ROBOFLOW_API_KEY",
    "PLAYER_WORKSPACE",
    "PLAYER_PROJECT",
    "PLAYER_VERSION",
    "LOGO_DETECTION_SOURCES",
    "PLAYER_EPOCHS",
    "LOGO_EPOCHS",
    "YOLO_IMGSZ",
    "PLAYER_NAME_HINTS",
    "IMAGE_EXTS",
    "YOLO_DATA_DIR",
    "YOLO_RUNS_DIR",
    "YOLO_WEIGHTS_DIR",
    "MERGED_LOGO_DIR",
    "PLAYER_WEIGHTS",
    "LOGO_WEIGHTS",
]
