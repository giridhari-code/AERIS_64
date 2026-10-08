"""Minimal BPE tokenizer (no external deps). Train on corpus, save/load JSON."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[str, str]] | None = None,
        vocab: dict[str, int] | None = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self.merges = list(merges or [])
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        if vocab:
            self.vocab = dict(vocab)
        else:
            self.vocab = {}
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), 1)

    @staticmethod
    def _word_to_chars(word: str) -> tuple[str, ...]:
        return tuple(list(word))

    @classmethod
    def train(cls, text: str, vocab_size: int = 500, min_freq: int = 2) -> "BPETokenizer":
        # byte/char base + word-level BPE merges
        specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>"]
        words = re.findall(r"\S+|\s+", text)
        word_freq = Counter(words)
        # start with character vocab
        vocab_set: set[str] = set(specials)
        for w in word_freq:
            vocab_set.update(list(w))
        splits: dict[str, tuple[str, ...]] = {w: cls._word_to_chars(w) for w in word_freq}

        merges: list[tuple[str, str]] = []
        while len(vocab_set) + len(merges) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for w, freq in word_freq.items():
                parts = splits[w]
                for i in range(len(parts) - 1):
                    pair_counts[(parts[i], parts[i + 1])] += freq
            if not pair_counts:
                break
            best, cnt = pair_counts.most_common(1)[0]
            if cnt < min_freq:
                break
            merges.append(best)
            a, b = best
            new_tok = a + b
            vocab_set.add(new_tok)
            for w in splits:
                parts = splits[w]
                if len(parts) < 2:
                    continue
                out: list[str] = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(parts[i])
                        i += 1
                splits[w] = tuple(out)

        vocab = {t: i for i, t in enumerate(sorted(vocab_set, key=lambda x: (x not in specials, x)))}
        # ensure specials first
        for i, t in enumerate(specials):
            vocab[t] = i
        # reindex rest
        others = sorted(t for t in vocab_set if t not in specials)
        for i, t in enumerate(others, start=len(specials)):
            vocab[t] = i
        return cls(merges=merges, vocab=vocab)

    def _get_pairs(self, parts: list[str]) -> set[tuple[str, str]]:
        return {(parts[i], parts[i + 1]) for i in range(len(parts) - 1)}

    def _bpe(self, token: str) -> list[str]:
        if not token:
            return []
        parts = list(token)
        if len(parts) == 1:
            return parts
        while True:
            pairs = self._get_pairs(parts)
            if not pairs:
                break
            ranked = [(self._merge_ranks.get(p, 10**9), p) for p in pairs]
            best_rank, best = min(ranked, key=lambda x: x[0])
            if best_rank == 10**9:
                break
            a, b = best
            new_parts: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    new_parts.append(a + b)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        unk = self.vocab.get(self.unk_token, 1)
        ids: list[int] = []
        if add_special:
            ids.append(self.vocab.get(self.bos_token, 2))
        for piece in re.findall(r"\S+|\s+", text):
            for tok in self._bpe(piece):
                ids.append(self.vocab.get(tok, unk))
        if add_special:
            ids.append(self.vocab.get(self.eos_token, 3))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = []
        for i in ids:
            t = self.id_to_token.get(int(i), self.unk_token)
            if t in (self.pad_token, self.bos_token, self.eos_token):
                continue
            toks.append(t)
        return "".join(toks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "bpe",
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data.get("merges") or []]
        return cls(
            merges=merges,
            vocab=data.get("vocab"),
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            bos_token=data.get("bos_token", "<bos>"),
            eos_token=data.get("eos_token", "<eos>"),
        )

    def to_vocab_dict(self) -> dict:
        return {
            "type": "bpe",
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
        }
