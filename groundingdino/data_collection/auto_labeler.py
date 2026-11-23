"""GroundingDINO joint labeler: person/player + logo/text in one pass.

Class ids in YOLO TXT:
  0 = person (keywords: person, player, man, people)
  1 = logo   (keywords: logo, text)

Writes bake-off outputs under comparison/groundingdino/:
  combined/  — all boxes on one image + merged TXT
  players/   — person boxes only (for side-by-side vs YOLO players)
  logos/     — logo/text boxes only (for side-by-side vs YOLO logos)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import groundingdino
import torch
import torchvision
from groundingdino.util.inference import load_image, load_model, predict

from label_io import draw_boxes_bgr, write_yolo_txt
from paths import (
    DINO_COMPARISON_DIR,
    DINO_WEIGHTS_HIGH,
    DINO_WEIGHTS_LOW,
    comparison_output_dirs,
    list_images,
    resolve_frames_dir,
)

CONFIG_PATH_LOW = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
CONFIG_PATH_HIGH = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinB_cfg.py"

# Phrase → (class_id, display_name). Order matters: more specific checked first.
PERSON_KEYWORDS = ("person", "player", "people", "man")
LOGO_KEYWORDS = ("logo", "text")

TEXT_PROMPT = "person . player . people . man . logo . text ."

BOX_THRESHOLD = 0.26
TEXT_THRESHOLD = 0.27
MIN_BOX_WIDTH = 0.015
MIN_BOX_HEIGHT = 0.015
MAX_BOX_WIDTH = 0.95
MAX_BOX_HEIGHT = 0.95

PERSON_COLOR = (255, 120, 0)  # BGR blue-ish
LOGO_COLOR = (0, 255, 0)  # BGR green


def resolve_class(detected_phrase: str):
    cleaned = detected_phrase.strip().lower()
    # Prefer logo/text first so "text" is never swallowed by unrelated matches.
    for kw in LOGO_KEYWORDS:
        if kw in cleaned:
            return 1, "logo" if kw == "logo" else "text"
    for kw in PERSON_KEYWORDS:
        if kw in cleaned:
            return 0, "person"
    return None, None


def clear_run_dirs(annotated_dir: Path, labels_dir: Path) -> None:
    for directory in (annotated_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def nms_keep(boxes: torch.Tensor, logits: torch.Tensor, iou_threshold: float = 0.5):
    if boxes.numel() == 0:
        return []
    corners = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    return torchvision.ops.nms(corners, logits, iou_threshold).tolist()


def auto_label_frames(
    frames_subdir: str,
    model_strength: str,
    limit: int | None = None,
    device: str = "cpu",
) -> None:
    frames_dir = resolve_frames_dir(frames_subdir)

    combined_ann, combined_lbl = comparison_output_dirs("groundingdino", "combined")
    players_ann, players_lbl = comparison_output_dirs("groundingdino", "players")
    logos_ann, logos_lbl = comparison_output_dirs("groundingdino", "logos")
    for ann, lbl in (
        (combined_ann, combined_lbl),
        (players_ann, players_lbl),
        (logos_ann, logos_lbl),
    ):
        clear_run_dirs(ann, lbl)

    if model_strength == "high":
        print("USING SWIN-B (HIGH)")
        model = load_model(str(CONFIG_PATH_HIGH), str(DINO_WEIGHTS_HIGH), device=device)
    else:
        print("USING SWIN-T (LOW)")
        model = load_model(str(CONFIG_PATH_LOW), str(DINO_WEIGHTS_LOW), device=device)

    images = list_images(frames_dir, limit=limit)
    print(f"Frames: {frames_dir} ({len(images)} images)")
    print(f"Prompt: {TEXT_PROMPT}")
    print(f"Output root: {DINO_COMPARISON_DIR}")
    print("Classes: 0=person (person/player/people/man), 1=logo (logo/text)")

    for img_path in images:
        image_source, image = load_image(str(img_path))
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        print(f"\nFile: {img_path.name}")
        print(f"Raw DINO Phrases: {phrases} | Confidence: {logits}")

        kept_boxes = []
        kept_logits = []
        kept_class_ids = []
        kept_names = []

        for box, logit, phrase in zip(boxes, logits, phrases):
            box_w, box_h = box[2].item(), box[3].item()
            if (
                box_w < MIN_BOX_WIDTH
                or box_h < MIN_BOX_HEIGHT
                or box_w > MAX_BOX_WIDTH
                or box_h > MAX_BOX_HEIGHT
            ):
                continue
            class_id, class_name = resolve_class(phrase)
            if class_id is None:
                continue
            kept_boxes.append(box)
            kept_logits.append(logit)
            kept_class_ids.append(class_id)
            kept_names.append(class_name)

        image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

        final_boxes = []
        final_logits = []
        final_ids = []
        final_names = []

        if kept_boxes:
            stacked = torch.stack(kept_boxes)
            stacked_logits = torch.stack(kept_logits)
            # Per-class NMS so player vs logo boxes do not suppress each other.
            for class_id in (0, 1):
                idxs = [i for i, cid in enumerate(kept_class_ids) if cid == class_id]
                if not idxs:
                    continue
                sub_boxes = stacked[idxs]
                sub_logits = stacked_logits[idxs]
                keep = nms_keep(sub_boxes, sub_logits, iou_threshold=0.5)
                for local_i in keep:
                    global_i = idxs[local_i]
                    final_boxes.append(stacked[global_i])
                    final_logits.append(stacked_logits[global_i])
                    final_ids.append(kept_class_ids[global_i])
                    final_names.append(kept_names[global_i])

        # Split by class for bake-off folders.
        person_boxes, person_labels, person_lines = [], [], []
        logo_boxes, logo_labels, logo_lines = [], [], []
        combined_boxes, combined_labels, combined_lines = [], [], []

        for box, logit, class_id, name in zip(final_boxes, final_logits, final_ids, final_names):
            cx, cy, bw, bh = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            label = f"{name} {logit.item():.2f}"
            tup = (cx, cy, bw, bh)
            combined_boxes.append(tup)
            combined_labels.append(label)
            combined_lines.append(line)
            if class_id == 0:
                person_boxes.append(tup)
                person_labels.append(label)
                person_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                logo_boxes.append(tup)
                logo_labels.append(label)
                logo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        stem_txt = f"{img_path.stem}.txt"
        name_jpg = img_path.name

        # Combined: draw persons then logos with different colors.
        combined_img = image_bgr.copy()
        if person_boxes:
            combined_img = draw_boxes_bgr(combined_img, person_boxes, person_labels, color=PERSON_COLOR)
        if logo_boxes:
            combined_img = draw_boxes_bgr(combined_img, logo_boxes, logo_labels, color=LOGO_COLOR)
        write_yolo_txt(combined_lbl / stem_txt, combined_lines)
        cv2.imwrite(str(combined_ann / name_jpg), combined_img)

        write_yolo_txt(players_lbl / stem_txt, person_lines)
        cv2.imwrite(
            str(players_ann / name_jpg),
            draw_boxes_bgr(image_bgr, person_boxes, person_labels, color=PERSON_COLOR),
        )

        write_yolo_txt(logos_lbl / stem_txt, logo_lines)
        cv2.imwrite(
            str(logos_ann / name_jpg),
            draw_boxes_bgr(image_bgr, logo_boxes, logo_labels, color=LOGO_COLOR),
        )

        print(
            f"Annotated {img_path.name}: "
            f"{len(person_lines)} person, {len(logo_lines)} logo/text "
            f"({len(combined_lines)} total)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundingDINO joint labeler (person/player + logo/text)."
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="Frames subdir under framed_videos, e.g. athletico_real_2023/close_ups",
    )
    parser.add_argument("--model", required=True, choices=["high", "low"])
    parser.add_argument("--limit", type=int, default=None, help="Max frames to process")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    auto_label_frames(args.directory, args.model, limit=args.limit, device=args.device)
