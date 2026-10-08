from __future__ import annotations

from typing import Iterable

from neurofield.tokenizer.specials import (
    ALL,
    decode_with_specials,
    encode_with_specials,
    ensure_specials_in_stoi,
    special_id_map,
    stop_token_ids,
)


class CharTokenizer:
    """Character-level tokenizer with optional chat special tokens."""

    def __init__(self, stoi: dict[str, int] | None = None, itos: dict[int, str] | None = None):
        self.stoi = dict(stoi or {})
        self.itos = {int(k): v for k, v in (itos or {}).items()}
        if self.stoi and not self.itos:
            self.itos = {i: c for c, i in self.stoi.items()}
        if self.itos and not self.stoi:
            self.stoi = {c: i for i, c in self.itos.items()}

    @classmethod
    def from_text(cls, text: str, with_specials: bool = True) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        if with_specials:
            stoi, itos = ensure_specials_in_stoi(stoi, itos)
        return cls(stoi, itos)

    @property
    def vocab_size(self) -> int:
        return max(len(self.stoi), 1)

    @property
    def special_ids(self) -> dict[str, int]:
        return special_id_map(self.stoi)

    @property
    def stop_ids(self) -> set[int]:
        return stop_token_ids(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi.get("?", self.stoi.get("<|unk|>", 0))
        if any(s in self.stoi for s in ALL):
            return encode_with_specials(text, self.stoi, unk_id=unk)
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return decode_with_specials(ids, self.itos)

    def to_vocab_dict(self) -> dict:
        return {
            "type": "char",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "specials": self.special_ids,
        }

    @classmethod
    def from_vocab_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d.get("stoi"), {int(k): v for k, v in (d.get("itos") or {}).items()})
