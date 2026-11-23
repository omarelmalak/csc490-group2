"""Before/after benchmark for jersey-synth logo augmentation.

1. Evaluate baseline logo weights on held-out synth valid set
2. Fine-tune from those weights on synth train (+ optional base train images)
3. Evaluate the new weights on the same synth valid set
4. Optionally re-check sponsor_detector valid (regression probe)

Example
-------
# Generate data first:
python synthesize_jersey_logos.py --n-samples 2000

# Baseline only (before):
python benchmark_jersey_synth.py --stage before \\
  --weights ../../../../weights/logo.pt

# Fine-tune + after:
python benchmark_jersey_synth.py --stage all \\
  --weights ../../../../weights/logo.pt \\
  --epochs 15 --device mps
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from config import (
    JERSEY_SYNTH_DIR,
    MERGED_LOGO_DIR,
    YOLO_DATA_DIR,
    YOLO_IMGSZ,
    YOLO_RUNS_DIR,
)

HERE = Path(__file__).resolve().parent
REPO_WEIGHTS_LOGO = HERE.parents[2] / "weights" / "logo.pt"


def run_eval(weights: Path, images: Path, labels: Path, out_json: Path, device: str, limit: int | None) -> dict:
    cmd = [
        sys.executable,
        str(HERE / "eval_logo_detector.py"),
        "--weights", str(weights),
        "--images", str(images),
        "--labels", str(labels),
        "--out", str(out_json),
        "--conf", "0.001",
        "--imgsz", str(YOLO_IMGSZ),
    ]
    if device:
        cmd += ["--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print("==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def summarize(tag: str, summary: dict) -> None:
    # eval_logo_detector writes keys — peek flexible
    print(f"\n=== {tag} ===")
    print(json.dumps(summary, indent=2)[:2000])


def build_finetune_yaml(
    synth_dir: Path,
    out_yaml: Path,
    include_base: bool,
    base_max: int,
) -> Path:
    """Train on synth train (+ optional sample of merged base train); val = synth valid."""
    train_dirs = [synth_dir / "train" / "images"]
    if include_base:
        base_img = MERGED_LOGO_DIR / "train" / "images"
        if base_img.exists():
            # Ultralytics accepts a txt file listing image paths
            list_path = out_yaml.parent / "finetune_train_list.txt"
            paths = sorted(base_img.glob("*"))
            paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            if base_max > 0:
                paths = paths[:base_max]
            synth_paths = sorted((synth_dir / "train" / "images").glob("*.jpg"))
            lines = [str(p.resolve()) for p in paths + synth_paths]
            list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            train_entry: str | Path = str(list_path.resolve())
        else:
            print(f"WARN: base train missing at {base_img}; synth-only fine-tune")
            train_entry = str(train_dirs[0].resolve())
    else:
        train_entry = str(train_dirs[0].resolve())

    data = {
        "train": train_entry,
        "val": str((synth_dir / "valid" / "images").resolve()),
        "nc": 1,
        "names": ["logo"],
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return out_yaml


def finetune(
    base_weights: Path,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    device: str,
    run_name: str,
) -> Path:
    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(YOLO_RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    best = YOLO_RUNS_DIR / run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Before/after jersey-synth logo benchmark.")
    parser.add_argument("--synth-dir", type=Path, default=JERSEY_SYNTH_DIR)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_WEIGHTS_LOGO if REPO_WEIGHTS_LOGO.exists() else None,
        help="Baseline logo detector weights (before)",
    )
    parser.add_argument(
        "--stage",
        choices=["before", "train", "after", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--include-base-train", action="store_true",
                        help="Mix in up to --base-max images from merged_logo_detector train")
    parser.add_argument("--base-max", type=int, default=3000)
    parser.add_argument("--run-name", default="logo_detector_jersey_synth_ft")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--also-sponsor-valid",
        action="store_true",
        help="Also eval on sponsor_detector/valid as a regression probe",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.weights is None or not args.weights.exists():
        raise FileNotFoundError(
            "Pass --weights to baseline logo.pt (repo weights/logo.pt or local-data best.pt)"
        )

    synth = args.synth_dir
    val_img = synth / "valid" / "images"
    val_lbl = synth / "valid" / "labels"
    if not val_img.exists():
        raise FileNotFoundError(
            f"Missing {val_img}. Run: python synthesize_jersey_logos.py --n-samples 2000"
        )

    out_dir = args.out_dir or (YOLO_DATA_DIR / "jersey_synth_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    before_json = out_dir / "before_synth_valid.json"
    after_json = out_dir / "after_synth_valid.json"
    ft_weights = YOLO_RUNS_DIR / args.run_name / "weights" / "best.pt"

    if args.stage in ("before", "all"):
        before = run_eval(args.weights, val_img, val_lbl, before_json, args.device, args.eval_limit)
        summarize("BEFORE (baseline on synth valid)", before)
        if args.also_sponsor_valid:
            sp_img = YOLO_DATA_DIR / "roboflow" / "sponsor_detector" / "valid" / "images"
            sp_lbl = YOLO_DATA_DIR / "roboflow" / "sponsor_detector" / "valid" / "labels"
            if sp_img.exists():
                sp = run_eval(
                    args.weights, sp_img, sp_lbl,
                    out_dir / "before_sponsor_valid.json",
                    args.device, args.eval_limit,
                )
                summarize("BEFORE (sponsor valid)", sp)

    if args.stage in ("train", "all"):
        data_yaml = build_finetune_yaml(
            synth, out_dir / "finetune_data.yaml",
            include_base=args.include_base_train,
            base_max=args.base_max,
        )
        ft_weights = finetune(
            args.weights, data_yaml, args.epochs, args.imgsz, args.device, args.run_name,
        )
        dest = out_dir / "logo_after_jersey_synth.pt"
        shutil.copy2(ft_weights, dest)
        print(f"Fine-tuned weights -> {ft_weights}")
        print(f"Copied -> {dest}")

    if args.stage in ("after", "all"):
        if not ft_weights.exists():
            raise FileNotFoundError(f"No fine-tuned weights at {ft_weights}; run --stage train first")
        after = run_eval(ft_weights, val_img, val_lbl, after_json, args.device, args.eval_limit)
        summarize("AFTER (fine-tuned on synth valid)", after)
        if args.also_sponsor_valid:
            sp_img = YOLO_DATA_DIR / "roboflow" / "sponsor_detector" / "valid" / "images"
            sp_lbl = YOLO_DATA_DIR / "roboflow" / "sponsor_detector" / "valid" / "labels"
            if sp_img.exists():
                sp = run_eval(
                    ft_weights, sp_img, sp_lbl,
                    out_dir / "after_sponsor_valid.json",
                    args.device, args.eval_limit,
                )
                summarize("AFTER (sponsor valid)", sp)

        if before_json.exists() and after_json.exists():
            b = json.loads(before_json.read_text(encoding="utf-8"))
            a = json.loads(after_json.read_text(encoding="utf-8"))
            comparison = {"before": b, "after": a}
            (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            print(f"\nWrote {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
