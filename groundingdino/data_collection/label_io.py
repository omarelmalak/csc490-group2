"""Shared helpers for writing YOLO TXT labels and annotated JPGs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_yolo_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    path.write_text(text, encoding="utf-8")


def draw_boxes_bgr(
    image_bgr: np.ndarray,
    boxes_cxcywh: list[tuple[float, float, float, float]],
    labels: list[str],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw normalized cxcywh boxes on a BGR image."""
    out = image_bgr.copy()
    img_h, img_w = out.shape[:2]
    for (cx, cy, bw, bh), label in zip(boxes_cxcywh, labels):
        x_min = int((cx - bw / 2) * img_w)
        y_min = int((cy - bh / 2) * img_h)
        x_max = int((cx + bw / 2) * img_w)
        y_max = int((cy + bh / 2) * img_h)
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2)
        font_scale = 0.4
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(out, (x_min, y_min - th - 6), (x_min + tw, y_min), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (x_min, y_min - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    return out


def xyxy_to_cxcywh_norm(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    bw = max(0.0, x2 - x1) / w
    bh = max(0.0, y2 - y1) / h
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    return cx, cy, bw, bh
