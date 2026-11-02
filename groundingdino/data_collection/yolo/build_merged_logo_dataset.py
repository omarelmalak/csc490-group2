"""Merge Roboflow logo detection sets into a 1-class (class-agnostic) YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from config import IMAGE_EXTS, MERGED_LOGO_DIR, YOLO_DATA_DIR


def iter_images(directory: Path):
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_dataset_as_class_agnostic(src_root: Path, dst_root: Path, prefix: str) -> dict:
    stats = {"images": 0, "labels": 0}
    for split in ("train", "valid", "test"):
        src_img = src_root / split / "images"
        if not src_img.exists():
            continue
        src_lbl = src_root / split / "labels"
        dst_img = ensure_dir(dst_root / split / "images")
        dst_lbl = ensure_dir(dst_root / split / "labels")

        for img_path in iter_images(src_img):
            new_stem = f"{prefix}__{img_path.stem}"
            shutil.copy2(img_path, dst_img / f"{new_stem}{img_path.suffix}")
            stats["images"] += 1

            label_path = src_lbl / f"{img_path.stem}.txt"
            out_lines = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = "0"
                        out_lines.append(" ".join(parts))
            (dst_lbl / f"{new_stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""),
                encoding="utf-8",
            )
            stats["labels"] += len(out_lines)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build class-agnostic merged logo dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=YOLO_DATA_DIR / "roboflow" / "manifest.json",
    )
    parser.add_argument("--reset", action="store_true", default=True)
    parser.add_argument("--no-reset", action="store_false", dest="reset")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Run download_datasets.py first. Missing {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    logo_sources = [(entry["name"], Path(entry["path"])) for entry in manifest["logos"]]

    if args.reset and MERGED_LOGO_DIR.exists():
        shutil.rmtree(MERGED_LOGO_DIR)
    ensure_dir(MERGED_LOGO_DIR)

    merge_stats = {}
    for name, root in logo_sources:
        merge_stats[name] = copy_dataset_as_class_agnostic(root, MERGED_LOGO_DIR, prefix=name)

    # Prefer valid split; fall back to train for val if needed.
    val_images = MERGED_LOGO_DIR / "valid" / "images"
    if not val_images.exists() or not any(val_images.iterdir()):
        val_images = MERGED_LOGO_DIR / "train" / "images"

    merged_yaml = {
        "train": str((MERGED_LOGO_DIR / "train" / "images").resolve()),
        "val": str(val_images.resolve()),
        "test": str((MERGED_LOGO_DIR / "test" / "images").resolve())
        if (MERGED_LOGO_DIR / "test" / "images").exists()
        else str(val_images.resolve()),
        "nc": 1,
        "names": ["logo"],
    }
    yaml_path = MERGED_LOGO_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_yaml, f, allow_unicode=True, sort_keys=False)

    print("Merged logo detector dataset:")
    for k, v in merge_stats.items():
        print(f"  - {k}: {v}")
    print("data.yaml ->", yaml_path)


if __name__ == "__main__":
    main()
