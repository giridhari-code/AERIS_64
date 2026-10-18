#!/usr/bin/env python3
"""Download public-domain Project Gutenberg plain text → data/pretrain_42m/en/

  python scripts/data/fetch_gutenberg_en.py
  python scripts/data/fetch_gutenberg_en.py --ids 1342,11,84,1661,2701 --max-books 20

Only use works that are public domain in *your* country. Default IDs are
well-known PD classics on Gutenberg. See https://www.gutenberg.org/policy/
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# Famous PD catalog numbers (verify on gutenberg.org if unsure)
DEFAULT_IDS = [
    11,      # Alice in Wonderland
    84,      # Frankenstein
    1342,    # Pride and Prejudice
    1661,    # Adventures of Sherlock Holmes
    2701,    # Moby Dick
    98,      # A Tale of Two Cities
    76,      # Huckleberry Finn
    74,      # Tom Sawyer
    174,     # Picture of Dorian Gray
    345,     # Dracula
    46,      # A Christmas Carol
    1400,    # Great Expectations
    1260,    # Jane Eyre
    43,      # Strange Case of Dr Jekyll and Mr Hyde
    16,      # Peter Pan
    55,      # Wizard of Oz
    120,     # Treasure Island
    2591,    # Grimms' Fairy Tales
    1232,    # The Prince (Machiavelli)
    2600,    # War and Peace (long)
]

START_MARKERS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
)
END_MARKERS = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
)


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = 0
    for i, line in enumerate(lines):
        u = line.upper()
        if any(m in u for m in START_MARKERS):
            start = i + 1
            break
    end = len(lines)
    for i in range(start, len(lines)):
        u = lines[i].upper()
        if any(m.upper() in lines[i] or m.upper() in u for m in END_MARKERS):
            end = i
            break
    body = "\n".join(lines[start:end]).strip()
    # collapse huge blank runs
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


def fetch_id(gid: int, timeout: float = 60.0) -> str:
    # Prefer UTF-8 plain text mirrors
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    ]
    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AERIS-42M-pretrain/1.0 (educational; public-domain fetch)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Gutenberg id={gid} failed: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch PD Gutenberg text into pretrain_42m/en")
    ap.add_argument("--out-dir", default="data/pretrain_42m/en")
    ap.add_argument("--ids", default="", help="Comma-separated Gutenberg IDs (default: curated list)")
    ap.add_argument("--max-books", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between downloads (be polite)")
    ap.add_argument("--min-chars", type=int, default=5000, help="Skip tiny/failed cleans")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.ids.strip():
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    else:
        ids = list(DEFAULT_IDS)
    ids = ids[: max(1, args.max_books)]

    ok = 0
    for gid in ids:
        dest = out / f"gutenberg_{gid}.txt"
        if dest.exists() and dest.stat().st_size > args.min_chars:
            print(f"skip exists {dest}")
            ok += 1
            continue
        try:
            print(f"fetch {gid} …")
            raw = fetch_id(gid)
            body = strip_gutenberg_boilerplate(raw)
            if len(body) < args.min_chars:
                print(f"  skip short after clean ({len(body)} chars)")
            else:
                dest.write_text(body, encoding="utf-8")
                words = len(body.split())
                print(f"  wrote {dest.name}  chars={len(body)}  words~={words}")
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {gid}: {e}")
        time.sleep(max(0.0, args.sleep))

    print(f"done books_ok={ok}/{len(ids)} → {out}")
    print("Next: list paths in data/pretrain_42m/SOURCES.md and run stats_pretrain_42m.py")


if __name__ == "__main__":
    main()
