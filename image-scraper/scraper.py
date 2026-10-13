"""
Download logo images for a list of brands from Bing Image Search.

Setup:
    pip install icrawler pillow imagehash

Usage:
    python scrape_logos.py                          # uses BRANDS below
    python scrape_logos.py --brands "Puma" "Emirates" "Etihad"
    python scrape_logos.py --brands-file brands.txt # one brand per line
    python scrape_logos.py --target 300 --out data/logos

Output:
    <out>/<brand>/<brand>_0001.jpg ...   (deduplicated, filtered)
    <out>/_raw/...                        (raw downloads, safe to delete)

Note: scraped logos are trademarked and the images are usually copyrighted.
Fine for coursework training, but keep the output folder out of the repo.
"""

import argparse
import shutil
from pathlib import Path

import imagehash
from icrawler.builtin import BingImageCrawler
from PIL import Image

# ---- Edit this list to change which logos get downloaded ----
BRANDS = [
    "Nike",
    "Adidas",
    "Spotify",
]

# Each brand is searched with every template below. More templates means
# more variety and more images surviving deduplication.
QUERY_TEMPLATES = [
    "{brand} logo current",
    "{brand} logo",
    "{brand} logo sign",
    "{brand} brand logo icon",
]

MIN_SIDE = 64          # skip images smaller than this (pixels)
HASH_THRESHOLD = 6     # perceptual hash distance; lower = stricter dedup
PER_QUERY_MAX = 150    # images requested per query (Bing caps around here)


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def crawl(query: str, dest: Path, max_num: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        downloader_threads=8,
        storage={"root_dir": str(dest)},
        log_level=40,  # errors only
    )
    crawler.crawl(keyword=query, max_num=max_num)


def collect(brand: str, raw_dir: Path, out_dir: Path, target: int) -> int:
    seen_hashes = []
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0

    for path in sorted(raw_dir.rglob("*")):
        if kept >= target:
            break
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.load()
                if min(img.size) < MIN_SIDE:
                    continue
                h = imagehash.phash(img.convert("RGB"))
        except Exception:
            continue  # corrupt or not an image

        if any(h - s <= HASH_THRESHOLD for s in seen_hashes):
            continue
        seen_hashes.append(h)

        kept += 1
        ext = path.suffix.lower() or ".jpg"
        shutil.copy(path, out_dir / f"{slug(brand)}_{kept:04d}{ext}")

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Download brand logo images.")
    parser.add_argument("--brands", nargs="+", help="Brand names to search for")
    parser.add_argument("--brands-file", help="Text file with one brand per line")
    parser.add_argument("--target", type=int, default=500, help="Images per brand")
    parser.add_argument("--out", default="logos", help="Output folder")
    args = parser.parse_args()

    if args.brands_file:
        lines = Path(args.brands_file).read_text().splitlines()
        brands = [b.strip() for b in lines if b.strip() and not b.startswith("#")]
    elif args.brands:
        brands = args.brands
    else:
        brands = BRANDS

    out = Path(args.out)
    for brand in brands:
        print(f"\n== {brand} ==")
        raw_dir = out / "_raw" / slug(brand)
        for template in QUERY_TEMPLATES:
            query = template.format(brand=brand)
            print(f"  searching: {query}")
            crawl(query, raw_dir / slug(query), PER_QUERY_MAX)

        kept = collect(brand, raw_dir, out / slug(brand), args.target)
        status = "" if kept >= args.target else "  (short; add more QUERY_TEMPLATES)"
        print(f"  kept {kept} unique images{status}")


if __name__ == "__main__":
    main()