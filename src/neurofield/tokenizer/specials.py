"""Chat / control special tokens for AERIS.

Training and serving should prefer these single-token IDs over regex on "User:" text.
Legacy char checkpoints without these IDs fall back to text stop heuristics.
"""

from __future__ import annotations

from typing import Iterable

# Canonical strings (also used in datasets)
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
PAD = "<|pad|>"
UNK = "<|unk|>"

ALL = (PAD, UNK, USER, ASSISTANT, END)

# Human-readable aliases still accepted when encoding datasets
ALIASES = {
    "User:": USER,
    "Assistant:": ASSISTANT,
    "user:": USER,
    "assistant:": ASSISTANT,
}


def ensure_specials_in_stoi(stoi: dict[str, int], itos: dict[int, str]) -> tuple[dict[str, int], dict[int, str]]:
    """Add missing special tokens at the end of a char vocab (new train only / expanded vocab)."""
    stoi = dict(stoi)
    itos = {int(k): v for k, v in itos.items()}
    for tok in ALL:
        if tok not in stoi:
            idx = max(itos.keys(), default=-1) + 1
            stoi[tok] = idx
            itos[idx] = tok
    return stoi, itos


def special_id_map(stoi: dict[str, int]) -> dict[str, int]:
    return {t: stoi[t] for t in ALL if t in stoi}


def encode_with_specials(text: str, stoi: dict[str, int], unk_id: int = 0) -> list[int]:
    """Encode text; emit a single id when a special or alias is matched."""
    # longest-first match
    needles = sorted(list(ALL) + list(ALIASES.keys()), key=len, reverse=True)
    ids: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for needle in needles:
            if text.startswith(needle, i):
                canon = ALIASES.get(needle, needle)
                if canon in stoi:
                    ids.append(stoi[canon])
                    i += len(needle)
                    matched = True
                    break
                # alias without vocab entry — fall through to chars
        if matched:
            continue
        ch = text[i]
        ids.append(stoi.get(ch, unk_id))
        i += 1
    return ids


def decode_with_specials(ids: Iterable[int], itos: dict[int, str]) -> str:
    parts = []
    for i in ids:
        parts.append(itos.get(int(i), "?"))
    return "".join(parts)


def stop_token_ids(stoi: dict[str, int]) -> set[int]:
    """Generation should stop when these appear (end or a new user turn)."""
    out: set[int] = set()
    for t in (END, USER):
        if t in stoi:
            out.add(stoi[t])
    return out
