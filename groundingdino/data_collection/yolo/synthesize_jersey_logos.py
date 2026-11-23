"""Paste brand logo assets onto soccer player boxes (from Untitled10_v3).

Builds a class-agnostic YOLO dataset (class 0 = logo) for jersey-domain
augmentation, e.g. Riyadh Air + Hyundai on player torsos.

Example
-------
python synthesize_jersey_logos.py \\
  --logos-root ../../../../local-data/.../data/input/brand_assets \\
  --player-root ../../../../local-data/.../data/yolo_train/roboflow/player_soccer \\
  --out ../../../../local-data/.../data/yolo_train/jersey_synth_riyadh_hyundai \\
  --n-samples 2000 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from config import BRAND_ASSETS_DIR, IMAGE_EXTS, JERSEY_SYNTH_DIR, YOLO_DATA_DIR

# Player soccer-iqjx5: 0 Arbitre, 1 Ballon, 2 Gardien, 3 Joueur
DEFAULT_PLAYER_CLASSES = {2, 3}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(directory: Path):
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def collect_logo_files(logos_root: Path) -> list[Path]:
    """Collect images from brand_assets/<brand>/*.jpg|png (or a flat folder)."""
    files: list[Path] = []
    if not logos_root.exists():
        raise FileNotFoundError(logos_root)
    # Prefer one subfolder per brand
    subdirs = [p for p in logos_root.iterdir() if p.is_dir()]
    roots = subdirs if subdirs else [logos_root]
    for root in roots:
        files.extend(iter_images(root))
    if not files:
        raise RuntimeError(f"No logo images under {logos_root}")
    return files


def read_player_boxes_xyxy(
    label_path: Path,
    width: int,
    height: int,
    keep_classes: set[int],
) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in keep_classes:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def load_logo_rgba(path: Path, max_side: int = 200):
    """RGB float32 + alpha/mask in 0..1 (v3 logic)."""
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    a = np.array(im)
    rgb = a[..., :3].astype(np.float32)
    alpha = a[..., 3].astype(np.float32) / 255.0
    if float(alpha.min()) > 0.98:
        whiteness = rgb.min(axis=2)
        mask = np.clip((250.0 - whiteness) / 250.0, 0.0, 1.0)
        if mask.mean() < 0.02:
            mask = np.ones_like(mask)
    else:
        mask = alpha
    mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    h, w = mask.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    return rgb, mask


def paste_logo(frame, rgb_logo, mask, box, rng, scale_range):
    """Paste warped logo onto torso band of a player box. Returns xyxy or None."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None
    tw = rng.uniform(*scale_range) * bw
    hh, ww = rgb_logo.shape[:2]
    th = tw * hh / max(ww, 1.0)
    if tw < 6 or th < 6:
        return None
    tw, th = int(tw), int(th)

    logo = cv2.resize(rgb_logo, (tw, th), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_AREA)

    pad = 0.25
    canvas_w, canvas_h = int(tw * (1 + 2 * pad)), int(th * (1 + 2 * pad))
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dx, dy = tw * pad, th * pad

    def j():
        return rng.uniform(-0.10, 0.10)

    dst = np.float32(
        [
            [dx + j() * tw, dy + j() * th],
            [dx + tw + j() * tw, dy + j() * th],
            [dx + tw + j() * tw, dy + th + j() * th],
            [dx + j() * tw, dy + th + j() * th],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    logo_w = cv2.warpPerspective(
        logo, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    mask_w = cv2.warpPerspective(
        m, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    if mask_w.max() < 0.05:
        return None

    tx1 = x1 + 0.10 * bw
    tx2 = max(tx1 + 1, x2 - 0.10 * bw - canvas_w)
    ty1 = y1 + 0.08 * bh
    ty2 = max(ty1 + 1, y1 + 0.60 * bh - canvas_h)
    px = int(rng.uniform(tx1, max(tx1, tx2)))
    py = int(rng.uniform(ty1, max(ty1, ty2)))
    if px < 0 or py < 0 or px + canvas_w >= W or py + canvas_h >= H:
        return None

    roi = frame[py : py + canvas_h, px : px + canvas_w].astype(np.float32)
    bg_mean = float(np.clip(roi.mean(), 20, 235))
    fg = logo_w[mask_w > 0.5]
    if fg.size == 0:
        return None
    logo_mean = float(np.clip(fg.mean(), 20, 235))
    gain = float(np.clip(bg_mean / logo_mean, 0.55, 1.8))
    logo_w = np.clip(logo_w * gain, 0, 255)

    a = (mask_w * rng.uniform(0.85, 1.0))[..., None]
    frame[py : py + canvas_h, px : px + canvas_w] = (logo_w * a + roi * (1 - a)).astype(np.uint8)

    ys, xs = np.where(mask_w > 0.35)
    if len(xs) == 0:
        return None
    bx1, bx2 = px + int(xs.min()), px + int(xs.max())
    by1, by2 = py + int(ys.min()), py + int(ys.max())
    return (bx1, by1, bx2, by2)


def synthesize(
    out_dir: Path,
    player_root: Path,
    logo_files: list[Path],
    n_samples: int,
    val_ratio: float,
    seed: int,
    scale_range: tuple[float, float],
    max_src_images: int,
    player_classes: set[int],
    reset: bool,
) -> dict:
    if reset and out_dir.exists():
        shutil.rmtree(out_dir)

    train_img = ensure_dir(out_dir / "train" / "images")
    train_lbl = ensure_dir(out_dir / "train" / "labels")
    val_img = ensure_dir(out_dir / "valid" / "images")
    val_lbl = ensure_dir(out_dir / "valid" / "labels")

    split_dir = player_root / "train"
    if not (split_dir / "images").exists():
        raise FileNotFoundError(f"Expected {player_root}/train/images")

    frames = list(iter_images(split_dir / "images"))[:max_src_images]
    if not frames:
        raise RuntimeError("No player frames found")

    rng = random.Random(seed)
    cache: dict[Path, tuple | None] = {}
    made = 0
    attempts = 0
    max_attempts = max(n_samples * 20, 1000)
    n_val = max(1, int(n_samples * val_ratio))

    while made < n_samples and attempts < max_attempts:
        attempts += 1
        frame_path = frames[rng.randrange(len(frames))]
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        pboxes = read_player_boxes_xyxy(
            split_dir / "labels" / f"{frame_path.stem}.txt",
            W, H, player_classes,
        )
        if not pboxes:
            continue

        labels = []
        for _ in range(rng.randint(1, 2)):
            lp = logo_files[rng.randrange(len(logo_files))]
            if lp not in cache:
                cache[lp] = load_logo_rgba(lp)
            item = cache[lp]
            if item is None:
                continue
            bb = paste_logo(img, item[0], item[1], rng.choice(pboxes), rng, scale_range)
            if bb:
                labels.append(bb)
        if not labels:
            continue

        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.2))
        if rng.random() < 0.6:
            img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

        # Last n_val samples → valid (held out for before/after)
        use_val = made >= (n_samples - n_val)
        img_dir, lbl_dir = (val_img, val_lbl) if use_val else (train_img, train_lbl)
        stem = f"synth__{made:06d}"
        cv2.imwrite(
            str(img_dir / f"{stem}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.uniform(35, 90))],
        )
        lines = []
        for bx1, by1, bx2, by2 in labels:
            cx = (bx1 + bx2) / 2.0 / W
            cy = (by1 + by2) / 2.0 / H
            bw = (bx2 - bx1) / W
            bh = (by2 - by1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        made += 1
        if made % 100 == 0:
            print(f"  synthesized {made}/{n_samples} (attempts={attempts})", flush=True)

    if made < n_samples:
        print(
            f"WARN: only produced {made}/{n_samples} samples after {attempts} attempts "
            f"(paste often fails on tiny player boxes / bad masks)"
        )

    data_yaml = {
        "train": str((out_dir / "train" / "images").resolve()),
        "val": str((out_dir / "valid" / "images").resolve()),
        "nc": 1,
        "names": ["logo"],
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    meta = {
        "n_requested": n_samples,
        "n_train": len(list(train_img.glob("*.jpg"))),
        "n_valid": len(list(val_img.glob("*.jpg"))),
        "n_logo_assets": len(logo_files),
        "player_root": str(player_root.resolve()),
        "scale_range": list(scale_range),
        "seed": seed,
        "data_yaml": str(yaml_path.resolve()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize jersey logo YOLO samples.")
    parser.add_argument(
        "--logos-root",
        type=Path,
        default=BRAND_ASSETS_DIR,
        help="Folder with brand subfolders of logo PNGs/JPGs",
    )
    parser.add_argument(
        "--player-root",
        type=Path,
        default=YOLO_DATA_DIR / "roboflow" / "player_soccer",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=JERSEY_SYNTH_DIR,
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-src-images", type=int, default=4000)
    parser.add_argument("--scale-min", type=float, default=0.08)
    parser.add_argument("--scale-max", type=float, default=0.28)
    parser.add_argument(
        "--player-classes",
        type=str,
        default="2,3",
        help="Comma-separated YOLO class ids to paste onto (default GK+player)",
    )
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    logo_files = collect_logo_files(args.logos_root)
    keep = {int(x) for x in args.player_classes.split(",") if x.strip() != ""}
    meta = synthesize(
        out_dir=args.out,
        player_root=args.player_root,
        logo_files=logo_files,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        scale_range=(args.scale_min, args.scale_max),
        max_src_images=args.max_src_images,
        player_classes=keep,
        reset=not args.no_reset,
    )
    print("Synthetic jersey dataset ready:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
