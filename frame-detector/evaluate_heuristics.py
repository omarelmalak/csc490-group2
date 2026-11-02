"""Evaluate the green-ratio and YOLO person-size frame heuristics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("close_up", "not_close_up", "uncertain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify labeled soccer frames and compare predictions with labels."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/soccernet_frames/spain_laliga/2016-2017/"
            "2016-08-21 - 21-15 Real Sociedad 0 - 3 Real Madrid/1"
        ),
        help="Directory containing close_up, not_close_up, and uncertain folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/frame-detector/heuristic_predictions.csv"),
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO weights.")
    parser.add_argument("--green-low", type=float, default=0.05)
    parser.add_argument("--green-high", type=float, default=0.85)
    parser.add_argument("--person-height", type=float, default=0.25)
    parser.add_argument("--person-confidence", type=float, default=0.50)
    parser.add_argument("--green-hue-low", type=int, default=35)
    parser.add_argument("--green-hue-high", type=int, default=85)
    parser.add_argument("--green-saturation-low", type=int, default=40)
    parser.add_argument("--green-value-low", type=int, default=40)
    return parser.parse_args()


def load_dependencies():
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "This evaluator requires opencv-python and ultralytics. Install them with:\n"
            "  python -m pip install opencv-python ultralytics"
        ) from error
    return cv2, YOLO


def green_ratio(image, cv2, args: argparse.Namespace) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (args.green_hue_low, args.green_saturation_low, args.green_value_low),
        (args.green_hue_high, 255, 255),
    )
    return float((mask > 0).mean())


def largest_person_height(
    results, frame_height: int, confidence_threshold: float
) -> tuple[int, float]:
    person_heights: list[float] = []
    for result in results:
        if result.boxes is None:
            continue
        for box, confidence, class_id in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            if int(class_id) != 0 or float(confidence) < confidence_threshold:
                continue
            person_heights.append((box[3] - box[1]) / frame_height)
    if not person_heights:
        return 0, 0.0
    return len(person_heights), max(person_heights)


def classify_frame(
    image_path: Path, model, cv2, args: argparse.Namespace
) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    frame_height = image.shape[0]
    ratio = green_ratio(image, cv2, args)
    results = model.predict(
        source=image,
        conf=args.person_confidence,
        classes=[0],
        verbose=False,
    )
    person_count, max_height = largest_person_height(
        results, frame_height, args.person_confidence
    )
    rejected_by_green = ratio < args.green_low or ratio > args.green_high
    prediction = (
        "not_close_up"
        if rejected_by_green or max_height < args.person_height
        else "close_up"
    )
    return {
        "green_ratio": ratio,
        "person_count": person_count,
        "max_person_height_ratio": max_height,
        "rejected_by_green": rejected_by_green,
        "prediction": prediction,
    }


def image_records(data_dir: Path):
    for label in LABELS:
        label_dir = data_dir / label
        if not label_dir.is_dir():
            raise SystemExit(f"Missing label directory: {label_dir}")
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield label, path


def write_results(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "label",
        "green_ratio",
        "person_count",
        "max_person_height_ratio",
        "rejected_by_green",
        "prediction",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(rows: list[dict[str, object]]) -> None:
    scored = [row for row in rows if row["label"] != "uncertain"]
    actual = Counter(row["label"] for row in scored)
    predicted = Counter(row["prediction"] for row in scored)
    true_positive = sum(
        row["label"] == "close_up" and row["prediction"] == "close_up"
        for row in scored
    )
    false_positive = sum(
        row["label"] == "not_close_up" and row["prediction"] == "close_up"
        for row in scored
    )
    false_negative = sum(
        row["label"] == "close_up" and row["prediction"] == "not_close_up"
        for row in scored
    )
    true_negative = sum(
        row["label"] == "not_close_up" and row["prediction"] == "not_close_up"
        for row in scored
    )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (true_positive + true_negative) / len(scored) if scored else 0

    print(f"Scored frames: {len(scored)} (excluded uncertain: {len(rows) - len(scored)})")
    print(f"Actual labels: {dict(actual)}")
    print(f"Predictions:   {dict(predicted)}")
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print("                 close_up  not_close_up")
    print(f"actual close_up  {true_positive:9d}  {false_negative:12d}")
    print(f"actual not_close {false_positive:9d}  {true_negative:12d}")
    print(f"\nAccuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")


def main() -> None:
    args = parse_args()
    cv2, YOLO = load_dependencies()
    records = list(image_records(args.data_dir))
    if not records:
        raise SystemExit(f"No supported images found under {args.data_dir}")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    rows: list[dict[str, object]] = []
    for index, (label, image_path) in enumerate(records, start=1):
        try:
            features = classify_frame(image_path, model, cv2, args)
        except Exception as error:
            print(f"Failed on {image_path}: {error}", file=sys.stderr)
            raise
        rows.append({"path": str(image_path), "label": label, **features})
        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} frames")

    write_results(rows, args.output_csv)
    print(f"\nWrote detailed results to {args.output_csv}")
    print_metrics(rows)


if __name__ == "__main__":
    main()
