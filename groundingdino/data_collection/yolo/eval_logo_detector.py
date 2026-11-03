"""Evaluate a class-agnostic YOLO logo detector on a labeled YOLO split.

The detector always predicts one class (``logo``), so all ground-truth classes are
mapped to class 0 before matching.  This is the same metric used to diagnose the
bake-off weights: it reports precision/recall at several confidence thresholds,
plus the confidence distribution of matched vs unmatched boxes.

Example
-------
python yolo/eval_logo_detector.py \
  --weights ../local-data/groundingdino/data_collection/weights/yolo/logo_detector/best.pt \
  --images Sponsor-detector-2/valid/images \
  --labels Sponsor-detector-2/valid/labels \
  --limit 125 --conf 0.001 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_gt_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Read YOLO detect/OBB labels as absolute xyxy boxes (class-agnostic)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(pred_boxes, pred_confs, gt_boxes, iou_thr=0.5):
    """Greedy one-to-one matching by descending confidence."""
    matched = 0
    if len(pred_boxes) == 0:
        return 0
    overlaps = iou_matrix(pred_boxes, gt_boxes)
    used = set()
    for i in np.argsort(-pred_confs):
        if len(gt_boxes) == 0:
            break
        j = int(np.argmax(overlaps[i]))
        if j not in used and overlaps[i, j] >= iou_thr:
            matched += 1
            used.add(j)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-agnostic YOLO logo detector evaluation.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold used to rank all detections.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a TP.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--save-annotated", type=Path, default=None)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    images = sorted(
        p for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No images found under {args.images}")

    model = YOLO(str(args.weights))
    if args.save_annotated:
        args.save_annotated.mkdir(parents=True, exist_ok=True)

    matched_confs: list[float] = []
    unmatched_confs: list[float] = []
    best_ious: list[float] = []
    per_image = []
    total_gt = 0
    total_pred = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = load_gt_boxes(args.labels / f"{image_path.stem}.txt", w, h)
        total_gt += len(gt)

        result = model(
            image, conf=args.conf, imgsz=args.imgsz,
            verbose=False, device=args.device, max_det=300,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0, 4))
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0,))
        total_pred += len(boxes)

        if len(boxes) and len(gt):
            overlaps = iou_matrix(boxes, gt)
            best_iou = overlaps.max(axis=1)
            for conf, iou in zip(confs, best_iou):
                best_ious.append(float(iou))
                if iou >= args.iou:
                    matched_confs.append(float(conf))
                else:
                    unmatched_confs.append(float(conf))
        else:
            unmatched_confs.extend(float(c) for c in confs)
            best_ious.extend([0.0] * len(boxes))

        per_image.append(
            {
                "image": image_path.name,
                "gt": int(len(gt)),
                "pred": int(len(boxes)),
                "max_conf": float(confs.max()) if len(confs) else 0.0,
            }
        )

        if args.save_annotated:
            annotated = image.copy()
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"logo {conf:.2f}", (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                )
            cv2.imwrite(str(args.save_annotated / image_path.name), annotated)

    matched = np.asarray(matched_confs)
    unmatched = np.asarray(unmatched_confs)
    best = np.asarray(best_ious)
    summary = {
        "weights": str(args.weights),
        "images": len(images),
        "gt_boxes": total_gt,
        "predictions_at_low_conf": total_pred,
        "matched_conf_median": float(np.median(matched)) if len(matched) else None,
        "matched_conf_p90": float(np.percentile(matched, 90)) if len(matched) else None,
        "unmatched_conf_median": float(np.median(unmatched)) if len(unmatched) else None,
        "unmatched_conf_p90": float(np.percentile(unmatched, 90)) if len(unmatched) else None,
        "best_iou_median": float(np.median(best)) if len(best) else None,
    }

    print(f"weights: {args.weights}")
    print(f"images: {len(images)} | GT boxes: {total_gt} | predictions@conf={args.conf}: {total_pred}")
    print(
        "matched conf: "
        f"n={len(matched)} median={summary['matched_conf_median']} p90={summary['matched_conf_p90']}"
    )
    print(
        "unmatched conf: "
        f"n={len(unmatched)} median={summary['unmatched_conf_median']} p90={summary['unmatched_conf_p90']}"
    )

    for threshold in (0.05, 0.1, 0.25, 0.5):
        tp = 0
        fp = 0
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            gt = load_gt_boxes(args.labels / f"{image_path.stem}.txt", w, h)
            result = model(
                image, conf=threshold, imgsz=args.imgsz,
                verbose=False, device=args.device, max_det=300,
            )[0]
            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None and len(result.boxes)
                else np.zeros((0, 4))
            )
            confs = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None and len(result.boxes)
                else np.zeros((0,))
            )
            matched_in_image = greedy_match(boxes, confs, gt, args.iou)
            tp += matched_in_image
            fp += len(boxes) - matched_in_image
        recall = tp / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary[f"conf_{threshold}"] = {
            "tp": tp, "fp": fp,
            "recall": recall, "precision": precision, "f1": f1,
        }
        print(
            f"conf>={threshold}: TP={tp} FP={fp} "
            f"recall={recall:.3f} precision={precision:.3f} f1={f1:.3f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
