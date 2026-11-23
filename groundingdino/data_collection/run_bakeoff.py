"""Orchestrate YOLO train (optional) + DINO/YOLO inference for the bake-off."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paths import LOGO_WEIGHTS, PLAYER_WEIGHTS

DC = Path(__file__).resolve().parent
YOLO_DIR = DC / "yolo"


def run(script: Path, extra: list[str] | None = None, cwd: Path | None = None) -> None:
    cmd = [sys.executable, str(script), *(extra or [])]
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or script.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector bake-off orchestrator.")
    parser.add_argument(
        "--directory",
        default="athletico_real_2023/close_ups",
        help="Frames subdir under framed_videos",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dino-model", choices=["high", "low"], default="low")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not download/train; require existing best.pt",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Download/train even if best.pt already exists",
    )
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-yolo-infer", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--player-epochs", type=int, default=None)
    parser.add_argument("--logo-epochs", type=int, default=None)
    args = parser.parse_args()

    weights_ready = PLAYER_WEIGHTS.exists() and LOGO_WEIGHTS.exists()
    should_train = args.force_train or (not args.skip_train and not weights_ready)

    if args.skip_train and not weights_ready:
        raise SystemExit(
            f"--skip-train set but weights missing:\n  {PLAYER_WEIGHTS}\n  {LOGO_WEIGHTS}"
        )

    if should_train:
        run(YOLO_DIR / "download_datasets.py", cwd=YOLO_DIR)
        run(YOLO_DIR / "build_merged_logo_dataset.py", cwd=YOLO_DIR)
        player_extra = []
        if args.player_epochs is not None:
            player_extra += ["--epochs", str(args.player_epochs)]
        run(YOLO_DIR / "train_player_yolo.py", player_extra, cwd=YOLO_DIR)
        logo_extra = []
        if args.logo_epochs is not None:
            logo_extra += ["--epochs", str(args.logo_epochs)]
        run(YOLO_DIR / "train_logo_yolo.py", logo_extra, cwd=YOLO_DIR)
    else:
        print(f"Using existing weights:\n  {PLAYER_WEIGHTS}\n  {LOGO_WEIGHTS}")

    if args.train_only:
        print("train-only: done.")
        return

    if not args.skip_dino:
        # Joint DINO labeler: person/player + logo/text in one pass
        # (writes combined/, players/, and logos/ under comparison/groundingdino/)
        run(
            DC / "auto_labeler.py",
            [
                "--directory", args.directory,
                "--model", args.dino_model,
                "--limit", str(args.limit),
            ],
            cwd=DC,
        )

    if not args.skip_yolo_infer:
        run(
            YOLO_DIR / "yolo_labeler_player.py",
            ["--directory", args.directory, "--limit", str(args.limit)],
            cwd=YOLO_DIR,
        )
        run(
            YOLO_DIR / "yolo_labeler_logo.py",
            ["--directory", args.directory, "--limit", str(args.limit)],
            cwd=YOLO_DIR,
        )

    print("\nBake-off outputs ready under data/output/comparison/{groundingdino,yolo}/")


if __name__ == "__main__":
    main()
