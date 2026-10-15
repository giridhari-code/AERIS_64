"""Eyes and hats — some hats require higher rarity."""

from __future__ import annotations

EYES = [
    ("dot", "."),
    ("round", "o"),
    ("wide", "O"),
    ("spark", "*"),
    ("line", "-"),
    ("plus", "+"),
]

# (id, glyph-ish label for hat line, min_rarity_rank)
# rarity rank: common=0 ... legendary=4
HATS = [
    ("none", "", 0),
    ("cap", "[cap]", 0),
    ("beanie", "[bn]", 0),
    ("bow", "~bow~", 1),  # uncommon+
    ("crown", "^V^", 3),  # epic+
    ("halo", " * ", 2),  # rare+
    ("flower", "@", 1),
    ("propeller", "-o-", 2),
]

RARITY_ORDER = ["common", "uncommon", "rare", "epic", "legendary"]


def rarity_rank(rarity: str) -> int:
    try:
        return RARITY_ORDER.index(rarity)
    except ValueError:
        return 0
