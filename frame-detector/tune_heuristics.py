"""Tune heuristic thresholds from cached frame features without rerunning YOLO."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune green and person-size thresholds on cached predictions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/frame-detector/heuristic_predictions.csv"),
    )
    parser.add_argument("--green-low-values", default="0.02,0.05,0.08,0.10")
    parser.add_argument("--green-high-values", default="0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--person-height-values", default="0.15,0.20,0.25,0.30,0.35")
    parser.add_argument(
        "--block-size",
        type=int,
        default=50,
        help="Consecutive frame IDs per temporal block.",
    )
    parser.add_argument(
        "--validation-block-mod",
        type=int,
        default=1,
        help="Block modulo assigned to validation.",
    )
    parser.add_argument(
        "--test-block-mod",
        type=int,
        default=2,
        help="Block modulo assigned to final test.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="F-beta weight; beta=2 emphasizes recall.",
    )
    return parser.parse_args()


def float_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def frame_id(path: str) -> int:
    try:
        return int(Path(path).stem)
    except ValueError as error:
        raise ValueError(f"Expected a numeric frame filename, got {path}") from error


def split_name(path: str, block_size: int, validation_mod: int, test_mod: int) -> str:
    block_mod = (frame_id(path) // block_size) % 5
    if block_mod == validation_mod:
        return "validation"
    if block_mod == test_mod:
        return "test"
    return "unused"


def predict(row: dict[str, str], green_low: float, green_high: float, height: float) -> str:
    green = float(row["green_ratio"])
    person_height = float(row["max_person_height_ratio"])
    if green < green_low or green > green_high or person_height < height:
        return "not_close_up"
    return "close_up"


def metrics(rows: list[dict[str, str]], thresholds: tuple[float, float, float], beta: float) -> dict[str, float]:
    green_low, green_high, height = thresholds
    scored = [row for row in rows if row["label"] != "uncertain"]
    tp = sum(row["label"] == "close_up" and predict(row, *thresholds) == "close_up" for row in scored)
    fp = sum(row["label"] == "not_close_up" and predict(row, *thresholds) == "close_up" for row in scored)
    fn = sum(row["label"] == "close_up" and predict(row, *thresholds) == "not_close_up" for row in scored)
    tn = sum(row["label"] == "not_close_up" and predict(row, *thresholds) == "not_close_up" for row in scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta_squared = beta * beta
    f_beta = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if beta_squared * precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / len(scored) if scored else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def format_metrics(values: dict[str, float]) -> str:
    return (
        f"accuracy={values['accuracy']:.3f}, precision={values['precision']:.3f}, "
        f"recall={values['recall']:.3f}, F={values['f_beta']:.3f}, "
        f"TP={int(values['tp'])}, FP={int(values['fp'])}, "
        f"FN={int(values['fn'])}, TN={int(values['tn'])}"
    )


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.validation_block_mod == args.test_block_mod:
        raise SystemExit("block-size must be positive and validation/test blocks must differ")
    if not 0 <= args.beta:
        raise SystemExit("beta must be non-negative")
    with args.input_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if row["label"] != "uncertain"]
    for row in rows:
        row["split"] = split_name(
            row["path"], args.block_size, args.validation_block_mod, args.test_block_mod
        )
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    if not validation or not test:
        raise SystemExit("The selected block split produced an empty validation or test set.")

    grid = (
        float_values(args.green_low_values),
        float_values(args.green_high_values),
        float_values(args.person_height_values),
    )
    best: tuple[float, tuple[float, float, float], dict[str, float]] | None = None
    for green_low in grid[0]:
        for green_high in grid[1]:
            if green_low >= green_high:
                continue
            for height in grid[2]:
                thresholds = (green_low, green_high, height)
                result = metrics(validation, thresholds, args.beta)
                candidate = (result["f_beta"], thresholds, result)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        raise SystemExit("No valid threshold combinations were provided.")

    _, thresholds, validation_result = best
    test_result = metrics(test, thresholds, args.beta)
    default = (0.05, 0.85, 0.25)
    print(f"Rows: validation={len(validation)}, test={len(test)}, beta={args.beta:g}")
    print(f"Block size: {args.block_size} frames")
    print(
        "Best validation thresholds: "
        f"green_low={thresholds[0]:g}, green_high={thresholds[1]:g}, "
        f"person_height={thresholds[2]:g}"
    )
    print(f"Validation: {format_metrics(validation_result)}")
    print(f"Test:       {format_metrics(test_result)}")
    print(f"Default validation: {format_metrics(metrics(validation, default, args.beta))}")
    print(f"Default test:       {format_metrics(metrics(test, default, args.beta))}")


if __name__ == "__main__":
    main()
