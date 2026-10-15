"""Thin wrapper over a HuggingFace `tokenizers` tokenizer.json (fast, Rust).

The in-repo BPE is pure python and fine for small corpora; for a 32k-vocab 1B
model use this (pip install tokenizers). Same interface as the other tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class HFTokenizer:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install tokenizers  (needed for HF tokenizer.json files)") from e
        self._tok = Tokenizer.from_file(str(path))

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special).ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=False)

    @staticmethod
    def is_hf_json(data: dict) -> bool:
        """HF tokenizer.json has top-level 'model' + ('added_tokens' | 'pre_tokenizer')."""
        return isinstance(data.get("model"), dict) and ("added_tokens" in data or "pre_tokenizer" in data)
