#!/usr/bin/env python3
"""Count chars / rough words under data/pretrain_42m for AERIS-42M."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("data/pretrain_42m")


def iter_txt(root: Path):
    if not root.exists():
        return
    for p in sorted(root.rglob("*.txt")):
        if p.is_file():
            yield p


def main() -> None:
    rows = []
    total_c = total_w = 0
    by_lang: dict[str, list[int]] = {}
    for p in iter_txt(ROOT):
        t = p.read_text(encoding="utf-8", errors="ignore")
        c = len(t)
        w = len(t.split())
        total_c += c
        total_w += w
        # path segment after pretrain_42m
        parts = p.relative_to(ROOT).parts
        lang = parts[0] if parts else "?"
        by_lang.setdefault(lang, [0, 0])
        by_lang[lang][0] += c
        by_lang[lang][1] += w
        rows.append((str(p), c, w))

    print(f"{'path':<55} {'chars':>12} {'words~':>12}")
    for path, c, w in rows:
        print(f"{path:<55} {c:12d} {w:12d}")
    print("-" * 80)
    for lang, (c, w) in sorted(by_lang.items()):
        print(f"lang={lang:<8} chars={c:>12d}  words~={w:>12d}")
    print("-" * 80)
    print(f"{'TOTAL':<55} {total_c:12d} {total_w:12d}")
    print()
    print("First pretrain aim: ~200M+ words across en/hi/bn (no code).")
    print(f"Current rough words: {total_w:,}  ({100.0 * total_w / 200_000_000:.4f}% of 200M target)")


if __name__ == "__main__":
    main()
