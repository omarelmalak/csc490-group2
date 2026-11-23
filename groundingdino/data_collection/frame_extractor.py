"""Extract match frames into wide_shots/ vs close_ups/.

Classification (unchanged from original):
  green_ratio > 0.40 → wide_shots
  else               → close_ups

Filenames are chronological by position in the video, e.g. frame_001245s.jpg
= 1245 seconds from start. Extracting windows out of order still sorts correctly
and re-extracting the same moment overwrites the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paths import FRAMED_VIDEOS_DIR, RAW_VIDEOS_DIR


def time_filename(frame_idx: int, fps: float) -> str:
    """Stable chronological name from absolute video frame index."""
    total_seconds = int(frame_idx / fps)
    # Include frame remainder so sub-second intervals do not collide.
    rem = int(frame_idx % max(fps, 1.0))
    return f"frame_{total_seconds:05d}s_{rem:02d}.jpg"


def extract_frames(
    video_file: str,
    duration_minutes: float,
    interval_seconds: float,
    start_minutes: float = 0.0,
) -> None:
    raw_video_path = RAW_VIDEOS_DIR / video_file
    if not raw_video_path.exists():
        alt = Path("data/input/raw_videos") / video_file
        if alt.exists():
            raw_video_path = alt
        else:
            raise FileNotFoundError(f"Video file not found at: {raw_video_path}")

    video_file_name = Path(video_file).stem
    output_dir_path = FRAMED_VIDEOS_DIR / video_file_name
    wide_dir_path = output_dir_path / "wide_shots"
    close_dir_path = output_dir_path / "close_ups"
    wide_dir_path.mkdir(parents=True, exist_ok=True)
    close_dir_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0:
        video_fps = 30.0

    start_frame = int(start_minutes * 60 * video_fps)
    end_frame = int((start_minutes + duration_minutes) * 60 * video_fps)
    end_frame = min(end_frame, total_video_frames)
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid window: start_frame={start_frame} end_frame={end_frame} "
            f"(video is {total_video_frames / (video_fps * 60):.1f} min)"
        )

    frame_step = max(1, int(round(interval_seconds * video_fps)))

    print(f"Video: {raw_video_path}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total Video Duration: {total_video_frames / (video_fps * 60):.2f} minutes")
    print(
        f"Window: {start_minutes:.1f} → {start_minutes + duration_minutes:.1f} min "
        f"(frames {start_frame}..{end_frame})"
    )
    print(f"Interval: every {interval_seconds}s (step {frame_step} frames)")
    print("Naming: chronological frame_{seconds:05d}s_{rem:02d}.jpg from video time")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_idx = start_frame
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    saved_count = 0

    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("Reached premature end of stream or unreadable frame.")
            break

        if (current_frame_idx - start_frame) % frame_step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

            frame_filename = time_filename(current_frame_idx, video_fps)

            # Original rule: lots of grass → wide; otherwise → close-up
            if green_ratio > 0.40:
                save_path = wide_dir_path / frame_filename
            else:
                save_path = close_dir_path / frame_filename

            cv2.imwrite(str(save_path), frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"Saved {saved_count} frames in this run... (last {frame_filename})")

        current_frame_idx += 1

    cap.release()
    print(f"Done. Extracted {saved_count} frames into '{output_dir_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract match frames; classify wide vs close-ups by grass ratio."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Filename under data/input/raw_videos/, e.g. athletico_real_2023.mp4",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=15.0,
        help="How many minutes of video to process from the start offset (default: 15).",
    )
    parser.add_argument(
        "--start-minutes",
        type=float,
        default=0.0,
        help="Skip this many minutes from the beginning before extracting (default: 0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between kept frames (default: 1.0).",
    )

    args = parser.parse_args()
    extract_frames(
        video_file=args.video,
        duration_minutes=args.minutes,
        interval_seconds=args.interval,
        start_minutes=args.start_minutes,
    )
