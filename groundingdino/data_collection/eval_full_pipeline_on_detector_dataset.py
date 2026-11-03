"""Evaluate the complete player -> logo -> containment -> doctor pipeline.

The dataset is the original logo-detector dataset (Sponsor-detector-2), which
already has player photos and jersey-logo boxes with brand ids.

Reported metrics
----------------
- detection_recall: matched logo boxes / all GT boxes
- containment_pass_rate: matched boxes inside a player box / all GT boxes
- doctor_oracle_acc: GT box crop -> Doctor accuracy
- doctor_pipeline_acc: detector crop -> Doctor accuracy (conditional on match+containment)
- end_to_end_acc: correct brand / all GT boxes

Example (Colab)
---------------
!python eval_full_pipeline_on_detector_dataset.py   --data-root /content/Sponsor-detector-2   --player-weights /content/drive/MyDrive/.../player_detector/best.pt   --logo-weights /content/drive/MyDrive/.../logo_detector/best.pt   --doctor-weights /content/drive/MyDrive/.../resnet50_jersey_brand_doctor.pth   --splits test valid
"""
# ============================================================
# Full-pipeline evaluation on the logo detector's original dataset
# Player Detector -> Logo Detector -> Containment -> Doctor
# ============================================================
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ------------------------- CLI CONFIG -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline evaluation: Player -> Logo -> Containment -> Doctor")
    p.add_argument("--data-root", default="/content/Sponsor-detector-2",
                   help="Sponsor-detector YOLO dataset root (contains data.yaml and test/valid splits).")
    p.add_argument("--player-weights", default="/content/drive/MyDrive/csc490/weights/yolo/player_detector/best.pt")
    p.add_argument("--logo-weights", default="/content/drive/MyDrive/csc490/weights/yolo/logo_detector/best.pt")
    p.add_argument("--doctor-weights", default="/content/drive/MyDrive/csc490/weights/resnet50_jersey_brand_doctor.pth")
    p.add_argument("--doctor-labels", default=None,
                   help="JSON label map if the doctor checkpoint is a plain state_dict.")
    p.add_argument("--splits", nargs="+", default=["test", "valid"])
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--containment-threshold", type=float, default=0.95)
    p.add_argument("--out-dir", default="/content/full_pipeline_eval")
    p.add_argument("--save-bad-cases", action="store_true", default=True)
    p.add_argument("--no-save-bad-cases", dest="save_bad_cases", action="store_false")
    return p.parse_args()

ARGS = parse_args()
DATA_ROOT = Path(ARGS.data_root)
PLAYER_WEIGHTS = Path(ARGS.player_weights)
LOGO_WEIGHTS = Path(ARGS.logo_weights)
DOCTOR_WEIGHTS = Path(ARGS.doctor_weights)
DOCTOR_LABEL_MAP = Path(ARGS.doctor_labels) if ARGS.doctor_labels else None
RUN_SPLITS = list(ARGS.splits)
DET_CONF = float(ARGS.det_conf)
MATCH_IOU = float(ARGS.match_iou)
CONTAINMENT_THRESHOLD = float(ARGS.containment_threshold)
SAVE_BAD_CASES = bool(ARGS.save_bad_cases)
OUT_DIR = Path(ARGS.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Dataset fallback: download Sponsor-detector-2 if not already present
if not (DATA_ROOT / "data.yaml").exists():
    from roboflow import Roboflow
    rf = Roboflow(api_key="A1UDb8W87l9RzmQ4CMlU")
    ds = rf.workspace("syed-fathaullah").project("sponsor-detector").version(2).download("yolov8")
    DATA_ROOT = Path(ds.location)
print("DATA_ROOT =", DATA_ROOT)

# 2) Logo weights fallback: the committed 25-epoch merged model
if not LOGO_WEIGHTS.exists():
    import urllib.request
    fallback = Path("/content/weights/logo_detector_best_25ep_merged.pt")
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        url = ("https://github.com/omarelmalak/csc490-group2/raw/"
               "omar-benchmark-groundingdino-yolo/groundingdino/data_collection/"
               "bakeoff_artifacts/logo_detector_best_25ep_merged.pt")
        urllib.request.urlretrieve(url, fallback)
    LOGO_WEIGHTS = fallback
print("LOGO_WEIGHTS =", LOGO_WEIGHTS)

# 3) Player fallback: COCO person model
if not PLAYER_WEIGHTS.exists():
    print("WARNING: player weights not found -> using yolov8n COCO person fallback")
    player_model = YOLO("yolov8n.pt")
    PLAYER_CLASS_IDS = [0]
else:
    player_model = YOLO(str(PLAYER_WEIGHTS))
    hints = {"joueur", "gardien", "player", "goalkeeper", "keeper"}
    PLAYER_CLASS_IDS = [
        idx for idx, name in player_model.names.items()
        if str(name).strip().lower() in hints
    ] or None
print("PLAYER_CLASS_IDS =", PLAYER_CLASS_IDS)

# 4) Logo detector + Doctor
logo_model = YOLO(str(LOGO_WEIGHTS))

if not DOCTOR_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Doctor checkpoint not found: {DOCTOR_WEIGHTS}\n"
        "Set DOCTOR_WEIGHTS to your trained resnet50_jersey_brand_doctor.pth."
    )

try:
    ckpt = torch.load(str(DOCTOR_WEIGHTS), map_location=DEVICE, weights_only=False)
except TypeError:
    ckpt = torch.load(str(DOCTOR_WEIGHTS), map_location=DEVICE)

if isinstance(ckpt, dict) and "model_state" in ckpt:
    doctor_state = ckpt["model_state"]
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
elif isinstance(ckpt, dict) and "idx_to_class" in ckpt:
    doctor_state = ckpt.get("state_dict", ckpt.get("model_state"))
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
else:
    doctor_state = ckpt
    if DOCTOR_LABEL_MAP is None or not Path(DOCTOR_LABEL_MAP).exists():
        raise ValueError("Checkpoint is a plain state_dict; set DOCTOR_LABEL_MAP to a JSON file.")
    label_map = json.loads(Path(DOCTOR_LABEL_MAP).read_text(encoding="utf-8"))
    idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}

doctor_model = resnet50(weights=None)
doctor_model.fc = nn.Linear(doctor_model.fc.in_features, len(idx_to_class))
doctor_model.load_state_dict(doctor_state)
doctor_model = doctor_model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Doctor classes:", len(idx_to_class))

# 5) Helpers
def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

DOCTOR_KEY_TO_IDX = {}
for idx, name in idx_to_class.items():
    DOCTOR_KEY_TO_IDX.setdefault(norm_name(name), idx)

def map_brand_to_doctor(brand):
    key = norm_name(brand)
    if key in DOCTOR_KEY_TO_IDX:
        return DOCTOR_KEY_TO_IDX[key]
    candidates = [
        (len(k), i) for k, i in DOCTOR_KEY_TO_IDX.items()
        if key and (key in k or k in key)
    ]
    if candidates:
        return sorted(candidates)[0][1]
    return None

def result_polygons(result):
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        return list(obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)), obb.conf.cpu().numpy()
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        polys, confs = [], []
        for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            confs.append(conf)
        return polys, np.asarray(confs)
    return [], np.asarray([])

def poly_to_xyxy(poly, w, h):
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return max(0, int(np.floor(x1))), max(0, int(np.floor(y1))), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))

def containment_ratio(inner_poly, outer_poly):
    inner = np.asarray(inner_poly, dtype=np.float32).reshape(-1, 2)
    outer = np.asarray(outer_poly, dtype=np.float32).reshape(-1, 2)
    inner_area = abs(cv2.contourArea(inner))
    if inner_area <= 1e-6:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(inner, outer)
    return float(inter_area / inner_area)

def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)

def read_gt_boxes(label_path, w, h, class_names):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            xs, ys = coords[0::2], coords[1::2]
            poly = np.array([[x * w, y * h] for x, y in zip(xs, ys)], dtype=np.float32)
            x1, y1, x2, y2 = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
        brand = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        boxes.append({
            "brand": brand,
            "class_id": cls_id,
            "poly": poly,
            "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
        })
    return boxes

@torch.no_grad()
def doctor_predict(image_rgb, poly):
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = poly_to_xyxy(poly, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    tensor = preprocess(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
    logits = doctor_model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(5, probs.numel()))
    return {
        "idx": int(topk.indices[0].item()),
        "brand": idx_to_class[int(topk.indices[0].item())],
        "conf": float(topk.values[0].item()),
        "top5": [
            {"brand": idx_to_class[int(i)], "conf": float(v)}
            for i, v in zip(topk.indices.tolist(), topk.values.tolist())
        ],
    }

def draw_bad_case(image_bgr, gt, pred, status, out_path):
    annotated = image_bgr.copy()
    gx1, gy1, gx2, gy2 = map(int, gt["xyxy"])
    cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(annotated, f"GT: {gt['brand']}", (gx1, max(gy1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if pred is not None:
        px1, py1, px2, py2 = map(int, pred["xyxy"])
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(annotated, f"Pred: {pred['brand']} {pred['conf']:.2f}",
                    (px1, max(py1 - 18, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(str(out_path), annotated)

# 6) Main evaluation
def evaluate_split(split):
    img_dir = DATA_ROOT / split / "images"
    lbl_dir = DATA_ROOT / split / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    data_cfg = yaml.safe_load((DATA_ROOT / "data.yaml").read_text(encoding="utf-8"))
    class_names = data_cfg.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names, key=lambda x: int(x))]

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    bad_dir = OUT_DIR / f"bad_cases_{split}"
    bad_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split,
        "images": len(images),
        "gt_total": 0,
        "gt_mapped_to_doctor": 0,
        "unmapped_gt": Counter(),
        "matched_total": 0,
        "contained_total": 0,
        "oracle_total": 0,
        "oracle_correct": 0,
        "pipeline_total": 0,
        "pipeline_correct": 0,
        "e2e_correct_all_gt": 0,
        "per_class": defaultdict(lambda: {"gt": 0, "oracle_correct": 0, "pipeline_total": 0, "pipeline_correct": 0}),
        "confusion": defaultdict(Counter),
        "records": [],
    }

    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        player_result = player_model(
            image_rgb, classes=PLAYER_CLASS_IDS, conf=0.25,
            verbose=False, device=YOLO_DEVICE,
        )[0]
        logo_result = logo_model(
            image_rgb, conf=DET_CONF, verbose=False, device=YOLO_DEVICE,
        )[0]

        player_polys, _ = result_polygons(player_result)
        logo_polys, logo_confs = result_polygons(logo_result)

        gt_boxes = read_gt_boxes(lbl_dir / f"{image_path.stem}.txt", w, h, class_names)
        stats["gt_total"] += len(gt_boxes)

        matched_pred_for_gt = {i: None for i in range(len(gt_boxes))}
        if len(logo_polys) and len(gt_boxes):
            pred_xyxy = [poly_to_xyxy(p, w, h) for p in logo_polys]
            iou_mat = np.zeros((len(pred_xyxy), len(gt_boxes)), dtype=np.float32)
            for i, pb in enumerate(pred_xyxy):
                for j, gt in enumerate(gt_boxes):
                    iou_mat[i, j] = iou_xyxy(pb, gt["xyxy"])
            used_gt = set()
            for pred_idx in np.argsort(-logo_confs):
                gt_idx = int(np.argmax(iou_mat[pred_idx]))
                if gt_idx not in used_gt and iou_mat[pred_idx, gt_idx] >= MATCH_IOU:
                    used_gt.add(gt_idx)
                    matched_pred_for_gt[gt_idx] = pred_idx

        for gt_idx, gt in enumerate(gt_boxes):
            gt_brand = gt["brand"]
            doctor_gt_idx = map_brand_to_doctor(gt_brand)
            stats["per_class"][gt_brand]["gt"] += 1

            if doctor_gt_idx is None:
                stats["unmapped_gt"][gt_brand] += 1
                stats["confusion"][gt_brand]["UNMAPPED"] += 1
                continue
            stats["gt_mapped_to_doctor"] += 1

            oracle = doctor_predict(image_rgb, gt["poly"])
            if oracle is not None:
                stats["oracle_total"] += 1
                if norm_name(oracle["brand"]) == norm_name(gt_brand):
                    stats["oracle_correct"] += 1
                    stats["per_class"][gt_brand]["oracle_correct"] += 1

            pred_idx = matched_pred_for_gt[gt_idx]
            if pred_idx is None:
                stats["confusion"][gt_brand]["MISSED"] += 1
                if SAVE_BAD_CASES:
                    draw_bad_case(image_bgr, gt, None, "MISSED", bad_dir / f"{image_path.stem}_{gt_idx}_missed.jpg")
                continue

            stats["matched_total"] += 1
            pred_poly = logo_polys[pred_idx]
            pred_conf = float(logo_confs[pred_idx])
            containment = max(
                [containment_ratio(pred_poly, p) for p in player_polys],
                default=0.0,
            )
            if containment < CONTAINMENT_THRESHOLD:
                stats["confusion"][gt_brand]["NO_CONTAINMENT"] += 1
                if SAVE_BAD_CASES:
                    bad_pred = {"xyxy": poly_to_xyxy(pred_poly, w, h), "brand": "?", "conf": pred_conf}
                    draw_bad_case(image_bgr, gt, bad_pred, "NO_CONTAINMENT",
                                  bad_dir / f"{image_path.stem}_{gt_idx}_containment.jpg")
                continue

            stats["contained_total"] += 1
            pipeline = doctor_predict(image_rgb, pred_poly)
            if pipeline is None:
                stats["confusion"][gt_brand]["BAD_CROP"] += 1
                continue

            stats["pipeline_total"] += 1
            stats["per_class"][gt_brand]["pipeline_total"] += 1
            correct = norm_name(pipeline["brand"]) == norm_name(gt_brand)
            if correct:
                stats["pipeline_correct"] += 1
                stats["e2e_correct_all_gt"] += 1
                stats["per_class"][gt_brand]["pipeline_correct"] += 1
            stats["confusion"][gt_brand][pipeline["brand"]] += 1

            stats["records"].append({
                "image": image_path.name,
                "gt_brand": gt_brand,
                "det_conf": pred_conf,
                "containment": containment,
                "doctor_brand": pipeline["brand"],
                "doctor_conf": pipeline["conf"],
                "correct": correct,
                "top5": pipeline["top5"],
            })

            if not correct and SAVE_BAD_CASES:
                bad_pred = {
                    "xyxy": poly_to_xyxy(pred_poly, w, h),
                    "brand": pipeline["brand"],
                    "conf": pipeline["conf"],
                }
                draw_bad_case(image_bgr, gt, bad_pred, "WRONG_BRAND",
                              bad_dir / f"{image_path.stem}_{gt_idx}_wrong.jpg")

    gt_total = stats["gt_total"]
    mapped = stats["gt_mapped_to_doctor"]
    stats["detection_recall"] = stats["matched_total"] / gt_total if gt_total else 0.0
    stats["containment_pass_rate"] = stats["contained_total"] / gt_total if gt_total else 0.0
    stats["doctor_oracle_acc"] = stats["oracle_correct"] / stats["oracle_total"] if stats["oracle_total"] else 0.0
    stats["doctor_pipeline_acc"] = stats["pipeline_correct"] / stats["pipeline_total"] if stats["pipeline_total"] else 0.0
    stats["end_to_end_acc_all_gt"] = stats["e2e_correct_all_gt"] / gt_total if gt_total else 0.0
    stats["end_to_end_acc_mapped_gt"] = stats["e2e_correct_all_gt"] / mapped if mapped else 0.0
    stats["unmapped_gt"] = dict(stats["unmapped_gt"])
    stats["per_class"] = {k: dict(v) for k, v in stats["per_class"].items()}
    stats["confusion"] = {k: dict(v) for k, v in stats["confusion"].items()}

    for brand, d in stats["per_class"].items():
        d["oracle_acc"] = d["oracle_correct"] / d["gt"] if d["gt"] else 0.0
        d["pipeline_acc"] = d["pipeline_correct"] / d["pipeline_total"] if d["pipeline_total"] else 0.0
        d["e2e_acc"] = d["pipeline_correct"] / d["gt"] if d["gt"] else 0.0

    print(f"\n===== {split.upper()} =====")
    print(f"images: {len(images)} | GT boxes: {gt_total} | GT mapped to Doctor vocab: {mapped}")
    if stats["unmapped_gt"]:
        print("WARNING: GT brands not found in Doctor vocab:", stats["unmapped_gt"])
    print(f"detector matched: {stats['matched_total']} ({stats['detection_recall']*100:.1f}%)")
    print(f"containment passed: {stats['contained_total']} ({stats['containment_pass_rate']*100:.1f}%)")
    print(f"Doctor oracle accuracy (GT crop): {stats['doctor_oracle_acc']*100:.1f}% "
          f"({stats['oracle_correct']}/{stats['oracle_total']})")
    print(f"Doctor pipeline accuracy (detector crop): {stats['doctor_pipeline_acc']*100:.1f}% "
          f"({stats['pipeline_correct']}/{stats['pipeline_total']})")
    print(f"End-to-end accuracy (all GT): {stats['end_to_end_acc_all_gt']*100:.1f}% "
          f"({stats['e2e_correct_all_gt']}/{gt_total})")
    print(f"End-to-end accuracy (mapped GT): {stats['end_to_end_acc_mapped_gt']*100:.1f}% "
          f"({stats['e2e_correct_all_gt']}/{mapped})")
    print("per class:")
    for brand, d in stats["per_class"].items():
        print(f"  {brand}: gt={d['gt']} oracle={d['oracle_acc']*100:.1f}% "
              f"pipeline={d['pipeline_acc']*100:.1f}% e2e={d['e2e_acc']*100:.1f}%")
    print("confusion:")
    for brand, counter in stats["confusion"].items():
        print(f"  {brand}: {dict(counter)}")

    summary = {k: v for k, v in stats.items() if k not in {"records", "per_class", "confusion"}}
    summary["per_class"] = stats["per_class"]
    summary["confusion"] = stats["confusion"]
    (OUT_DIR / f"summary_{split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT_DIR / f"records_{split}.jsonl", "w", encoding="utf-8") as f:
        for rec in stats["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("saved:", OUT_DIR / f"summary_{split}.json")
    return stats


if __name__ == "__main__":
    all_stats = {}
    for split in RUN_SPLITS:
        if (DATA_ROOT / split / "images").exists():
            all_stats[split] = evaluate_split(split)
        else:
            print("skip missing split:", split)
