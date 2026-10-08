#!/usr/bin/env python3
"""
Common Crawl / FineWeb sample helper — DOES NOT download full Common Crawl.

Options:
  1) Print how to list one CC WET path index (tiny download)
  2) Optionally stream a few FineWeb rows if `datasets` is installed

Full CC is multi-tiabyte per month. Use FineWeb samples for local experiments.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request

CC_WET_PATHS_EXAMPLE = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-39/wet.paths.gz"
)


def show_cc_paths(n: int = 5) -> None:
    print(f"Fetching path list index (small): {CC_WET_PATHS_EXAMPLE}")
    try:
        with urllib.request.urlopen(CC_WET_PATHS_EXAMPLE, timeout=60) as resp:
            raw = resp.read()
        lines = gzip.decompress(raw).decode("utf-8", errors="replace").splitlines()
        print(f"Total WET parts in this crawl listing: {len(lines)}")
        print("First paths (each part is still a large .warc.wet.gz file):")
        for line in lines[:n]:
            print(f"  https://data.commoncrawl.org/{line}")
        print("\nWARNING: downloading all parts = many terabytes. Do not wget -r.")
    except Exception as e:
        print(f"Could not fetch paths (network/firewall): {e}", file=sys.stderr)
        print("See docs/COMMON_CRAWL.md for manual URLs.")


def stream_fineweb(limit: int = 20) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        print("Then: load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', streaming=True)")
        return
    print(f"Streaming up to {limit} FineWeb sample-10BT documents...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for i, row in enumerate(ds):
        text = (row.get("text") or "")[:200].replace("\n", " ")
        print(f"[{i}] {text}...")
        if i + 1 >= limit:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="CC / FineWeb sample helper (not full CC)")
    ap.add_argument("--show-cc-paths", action="store_true", help="List first WET paths of one crawl")
    ap.add_argument("--stream-fineweb", action="store_true", help="Stream a few FineWeb rows")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    if not args.show_cc_paths and not args.stream_fineweb:
        ap.print_help()
        print("\nRead docs/COMMON_CRAWL.md — full Common Crawl is not downloaded by this script.")
        return
    if args.show_cc_paths:
        show_cc_paths(args.limit)
    if args.stream_fineweb:
        stream_fineweb(args.limit)


if __name__ == "__main__":
    main()
