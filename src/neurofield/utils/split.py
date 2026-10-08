"""Leak-free train/validation splitting for text corpora (pure python)."""

from __future__ import annotations


def split_blocks(text: str, val_frac: float = 0.1) -> tuple[str, str, dict[str, int]]:
    """Deterministic, de-duplicated train/val split on blank-line (or line) blocks.

    De-duplication happens BEFORE the split so an identical block cannot sit in
    both sides (that was a second leak: the corpus repeats lines).
    """
    raw = text.split("\n\n") if "\n\n" in text else text.split("\n")
    blocks = [b.strip() for b in raw if b.strip()]
    unique = list(dict.fromkeys(blocks))
    every = max(2, round(1.0 / max(val_frac, 1e-6)))
    val = [b for i, b in enumerate(unique) if i % every == every - 1]
    train = [b for i, b in enumerate(unique) if i % every != every - 1]
    sep = "\n\n" if "\n\n" in text else "\n"
    stats = {"blocks": len(blocks), "unique": len(unique), "train": len(train), "val": len(val)}
    return sep.join(train) + "\n", sep.join(val) + "\n", stats
