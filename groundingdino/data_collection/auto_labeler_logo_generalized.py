"""Deprecated shim — use auto_labeler.py (joint person + logo/text detector)."""

from __future__ import annotations

import argparse
import sys
import warnings

from auto_labeler import auto_label_frames

warnings.warn(
    "auto_labeler_logo_generalized.py is deprecated. "
    "Use auto_labeler.py which detects person/player/people/man + logo/text in one pass.",
    DeprecationWarning,
    stacklevel=1,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deprecated: redirects to joint auto_labeler.py"
    )
    parser.add_argument("--directory", required=True)
    parser.add_argument("--model", required=True, choices=["high", "low"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--source",
        choices=["framed", "logos"],
        default="framed",
        help="Ignored; joint labeler always reads framed_videos (or an absolute path).",
    )
    args = parser.parse_args()
    print(
        "NOTE: running joint auto_labeler (person + logo/text). "
        "Separate logo-only script is no longer needed.",
        file=sys.stderr,
    )
    auto_label_frames(args.directory, args.model, limit=args.limit, device=args.device)
