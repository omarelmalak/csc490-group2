"""Shared roots for bake-off scripts.

Prefer the workspace local-data tree (weights + framed videos already live there).
Fall back to this package directory if local-data is missing.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_local_data_collection() -> Path | None:
    """Walk up from this file looking for local-data/groundingdino/data_collection."""
    for parent in [_PACKAGE_DIR, *_PACKAGE_DIR.parents]:
        candidate = parent / "local-data" / "groundingdino" / "data_collection"
        if candidate.is_dir():
            return candidate
        # Also handle workspace root one level above the git repo (csc490/csc490-group2/...)
        sibling = parent.parent / "local-data" / "groundingdino" / "data_collection"
        if sibling.is_dir():
            return sibling
    return None


_LOCAL_DC = _find_local_data_collection()
DATA_COLLECTION_ROOT = _LOCAL_DC if _LOCAL_DC is not None else _PACKAGE_DIR

DATA_DIR = DATA_COLLECTION_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
WEIGHTS_DIR = DATA_COLLECTION_ROOT / "weights"

FRAMED_VIDEOS_DIR = INPUT_DIR / "framed_videos"
RAW_VIDEOS_DIR = INPUT_DIR / "raw_videos"
LOGOS_INPUT_DIR = INPUT_DIR / "logos"

COMPARISON_DIR = OUTPUT_DIR / "comparison"
DINO_COMPARISON_DIR = COMPARISON_DIR / "groundingdino"
YOLO_COMPARISON_DIR = COMPARISON_DIR / "yolo"

YOLO_WEIGHTS_DIR = WEIGHTS_DIR / "yolo"
YOLO_DATA_DIR = DATA_DIR / "yolo_train"
MERGED_LOGO_DIR = YOLO_DATA_DIR / "merged_logo_detector"
YOLO_RUNS_DIR = YOLO_DATA_DIR / "runs"

PLAYER_WEIGHTS = YOLO_WEIGHTS_DIR / "player_detector" / "best.pt"
LOGO_WEIGHTS = YOLO_WEIGHTS_DIR / "logo_detector" / "best.pt"

DINO_WEIGHTS_LOW = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
DINO_WEIGHTS_HIGH = WEIGHTS_DIR / "groundingdino_swinb_cogcoor.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_frames_dir(directory: str) -> Path:
    """Resolve a frames subdirectory under framed_videos, or an absolute/relative path."""
    raw = Path(directory)
    if raw.is_absolute() and raw.exists():
        return raw
    under_framed = FRAMED_VIDEOS_DIR / directory
    if under_framed.exists():
        return under_framed
    under_cwd = Path.cwd() / directory
    if under_cwd.exists():
        return under_cwd
    raise FileNotFoundError(
        f"Frames directory not found: {directory!r} "
        f"(looked under {FRAMED_VIDEOS_DIR} and cwd)"
    )


def list_images(directory: Path, limit: int | None = None) -> list[Path]:
    images = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        images = images[: max(0, limit)]
    return images


def comparison_output_dirs(strategy: str, kind: str) -> tuple[Path, Path]:
    """Return (annotated_frames_dir, txt_labels_dir) for a bake-off run."""
    root = COMPARISON_DIR / strategy / kind
    annotated = root / "annotated_frames"
    labels = root / "txt_labels"
    annotated.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return annotated, labels
