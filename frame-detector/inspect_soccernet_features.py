"""Inspect SoccerNet SN-Features without downloading the full dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


REPO = "SoccerNet/SN-Features"
BRANCHES = (
    "baidu-soccer-embeddings",
    "resnet-tf2",
    "resnet-tf2-pca512",
    "player-boundingbox-maskrcnn",
    "field-calib-ccbv",
)
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/tree"


def list_tree(revision: str, path: str = "") -> list[dict]:
    encoded_path = quote(path, safe="/")
    url = f"{API_ROOT}/{revision}/{encoded_path}?recursive=false&limit=100"
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def first_game(revision: str) -> str:
    leagues = [item["path"] for item in list_tree(revision) if item["type"] == "directory"]
    if not leagues:
        raise RuntimeError(f"No league directories found on branch {revision!r}.")

    seasons = [
        item["path"]
        for item in list_tree(revision, leagues[0])
        if item["type"] == "directory"
    ]
    if not seasons:
        raise RuntimeError(f"No seasons found under {leagues[0]!r}.")

    games = [
        item["path"]
        for item in list_tree(revision, seasons[0])
        if item["type"] == "directory"
    ]
    if not games:
        raise RuntimeError(f"No games found under {seasons[0]!r}.")
    return games[0]


def download_game(game_path: str, output_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "Downloading requires huggingface_hub. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from error

    local_path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision="player-boundingbox-maskrcnn",
        local_dir=output_dir,
        allow_patterns=[f"{game_path}/*_player_boundingbox_maskrcnn.json"],
    )
    print(f"Downloaded metadata to {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-game",
        metavar="PATH",
        help="Download one game's player-box JSON, for example "
        "'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/soccernet-features"),
    )
    args = parser.parse_args()

    print(f"{REPO} contains separate branches:")
    for branch in BRANCHES:
        print(f"  - {branch}")

    branch = "player-boundingbox-maskrcnn"
    game = args.download_game or first_game(branch)
    print(f"\nExample game on {branch}: {game}")
    print(
        "\nThese files contain model metadata, not RGB frames. "
        "They cannot support HSV filtering or image labeling."
    )

    if args.download_game:
        download_game(args.download_game, args.output_dir)


if __name__ == "__main__":
    main()
