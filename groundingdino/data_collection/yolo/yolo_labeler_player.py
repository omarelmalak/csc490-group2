"""YOLO player bake-off labeler — mirrors auto_labeler.py IO."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

_DC = Path(__file__).resolve().parent.parent
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from label_io import draw_boxes_bgr, write_yolo_txt, xyxy_to_cxcywh_norm  # noqa: E402
from paths import (  # noqa: E402
    PLAYER_WEIGHTS,
    YOLO_COMPARISON_DIR,
    comparison_output_dirs,
    list_images,
    resolve_frames_dir,
)
from yolo.config import PLAYER_NAME_HINTS  # noqa: E402


def clear_run_dirs(annotated_dir: Path, labels_dir: Path) -> None:
    for directory in (annotated_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO player bake-off labeler.")
    parser.add_argument(
        "--directory",
        required=True,
        help="Frames subdir under framed_videos, e.g. athletico_real_2023/close_ups",
    )
    parser.add_argument("--weights", type=Path, default=PLAYER_WEIGHTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Player weights not found: {args.weights}. Train with train_player_yolo.py first."
        )

    frames_dir = resolve_frames_dir(args.directory)
    annotated_dir, labels_dir = comparison_output_dirs("yolo", "players")
    clear_run_dirs(annotated_dir, labels_dir)

    model = YOLO(str(args.weights))
    class_ids = [
        idx
        for idx, name in model.names.items()
        if str(name).strip().lower() in PLAYER_NAME_HINTS
    ] or None

    images = list_images(frames_dir, limit=args.limit)
    print(f"Frames: {frames_dir} ({len(images)} images)")
    print(f"Weights: {args.weights}")
    print(f"Player class ids: {class_ids}")
    print(f"Output: {YOLO_COMPARISON_DIR / 'players'}")

    for img_path in images:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Skip unreadable {img_path}")
            continue
        h, w = image_bgr.shape[:2]
        result = model(
            image_bgr,
            classes=class_ids,
            conf=args.conf,
            verbose=False,
            device=args.device,
        )[0]

        lines: list[str] = []
        boxes_norm = []
        labels = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(float, box)
                cx, cy, bw, bh = xyxy_to_cxcywh_norm(x1, y1, x2, y2, w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                boxes_norm.append((cx, cy, bw, bh))
                name = model.names.get(int(cls_id), "player")
                labels.append(f"{name} {float(conf):.2f}")

        write_yolo_txt(labels_dir / f"{img_path.stem}.txt", lines)
        annotated = draw_boxes_bgr(image_bgr, boxes_norm, labels, color=(255, 120, 0))
        cv2.imwrite(str(annotated_dir / img_path.name), annotated)
        print(f"{img_path.name}: {len(lines)} players")


if __name__ == "__main__":
    main()
