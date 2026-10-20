import argparse
import os
import cv2
import numpy as np


def extract_frames(video_file, duration_minutes, interval_seconds):
    raw_video_path = f"data/input/raw_videos/{video_file}"
    if not os.path.exists(raw_video_path):
        raise FileNotFoundError(f"Video file not found at: {raw_video_path}")

    video_file_name = video_file.split(".")[0]
    output_dir_path = f"data/input/framed_videos/{video_file_name}"

    # Create the sorted subdirectories
    wide_dir_path = os.path.join(output_dir_path, "wide_shots")
    close_dir_path = os.path.join(output_dir_path, "close_ups")
    os.makedirs(wide_dir_path, exist_ok=True)
    os.makedirs(close_dir_path, exist_ok=True)

    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 30.0

    # Fixed syntax error here
    max_frames_to_read = int(duration_minutes * 60 * video_fps)
    max_frames_to_read = min(max_frames_to_read, total_video_frames)
    frame_step = max(1, int(round(interval_seconds * video_fps)))

    expected_saved_frames = (max_frames_to_read + frame_step - 1) // frame_step

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total Video Duration: {total_video_frames / (video_fps * 60):.2f} minutes")
    print(f"Target Duration: First {duration_minutes} minutes")
    print(f"Processing up to frame index {max_frames_to_read} with step {frame_step}")
    print(f"Estimated output frames: {expected_saved_frames}")

    # Define color boundaries for the pitch once before the loop
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    current_frame_idx = 0
    saved_count = 0

    while current_frame_idx < max_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            print("Reached premature end of stream or unreadable frame.")
            break

        if current_frame_idx % frame_step == 0:
            # Calculate green grass ratio
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

            frame_filename = f"frame_{saved_count:05d}.jpg"

            # Route to the appropriate folder based on classification
            if green_ratio > 0.40:
                save_path = os.path.join(wide_dir_path, frame_filename)
            else:
                save_path = os.path.join(close_dir_path, frame_filename)

            cv2.imwrite(save_path, frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"Extracted {saved_count} frames...")

        current_frame_idx += 1

    cap.release()
    print(f"Done. Extracted {saved_count} frames into '{output_dir_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from the first X minutes of a match video."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the source match video file.",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=15.0,
        help="Number of minutes to extract from the start of the video (default: 15.0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Capture interval in seconds between extracted frames (default: 1.0).",
    )

    args = parser.parse_args()
    extract_frames(
        video_file=args.video,
        duration_minutes=args.minutes,
        interval_seconds=args.interval,
    )