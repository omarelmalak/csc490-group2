"""Download Roboflow player + logo detection datasets for YOLO training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from config import (
    LOGO_DETECTION_SOURCES,
    PLAYER_PROJECT,
    PLAYER_VERSION,
    PLAYER_WORKSPACE,
    ROBOFLOW_API_KEY,
    YOLO_DATA_DIR,
)


def _copy_or_link_dataset(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow YOLO datasets.")
    parser.add_argument(
        "--out",
        type=Path,
        default=YOLO_DATA_DIR / "roboflow",
        help="Directory to store downloaded datasets",
    )
    args = parser.parse_args()

    from roboflow import Roboflow

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    manifest: dict = {"player": None, "logos": []}

    print("Downloading player dataset...")
    player_ds = (
        rf.workspace(PLAYER_WORKSPACE)
        .project(PLAYER_PROJECT)
        .version(PLAYER_VERSION)
        .download("yolov8")
    )
    player_src = Path(getattr(player_ds, "location", "soccer-iqjx5-2"))
    player_dst = _copy_or_link_dataset(player_src, out / "player_soccer")
    manifest["player"] = str(player_dst.resolve())
    print(f"Player dataset -> {player_dst}")

    for name, workspace, project, version in LOGO_DETECTION_SOURCES:
        print(f"Downloading logo dataset {name}...")
        dataset = rf.workspace(workspace).project(project).version(version).download("yolov8")
        src = Path(dataset.location)
        dst = _copy_or_link_dataset(src, out / name)
        manifest["logos"].append({"name": name, "path": str(dst.resolve())})
        print(f"[{name}] -> {dst}")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
