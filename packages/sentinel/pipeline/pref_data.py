"""Preference / SFT data loaders (jsonl + text). Schema normalization only — no demo data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

EncodeFn = Callable[[str], list[int]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_pref_row(row: dict[str, Any]) -> dict[str, str]:
    """Map common industry schemas → {prompt, chosen, rejected}."""
    prompt = (
        row.get("prompt")
        or row.get("user")
        or row.get("question")
        or row.get("instruction")
        or ""
    )
    chosen = row.get("chosen") or row.get("response_chosen") or row.get("answer") or ""
    rejected = row.get("rejected") or row.get("response_rejected") or row.get("bad") or ""
    return {
        "prompt": str(prompt),
        "chosen": str(chosen),
        "rejected": str(rejected),
    }


def normalize_sft_row(row: dict[str, Any]) -> dict[str, str]:
    if "text" in row:
        return {"text": str(row["text"])}
    prompt = row.get("prompt") or row.get("instruction") or row.get("user") or ""
    response = row.get("response") or row.get("output") or row.get("assistant") or ""
    if prompt and response:
        return {"text": f"User: {prompt}\nAssistant: {response}"}
    return {"text": str(row.get("content") or row.get("body") or "")}


def build_encode_fn(
    tokenizer_json: str | Path | None = None,
    vocab_size: int = 32000,
) -> EncodeFn:
    """Production encode: HF tokenizer.json. No silent demo corpus paths."""
    if tokenizer_json:
        path = Path(tokenizer_json)
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer not found: {path}")
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(path))

            def encode(text: str) -> list[int]:
                return list(tok.encode(text).ids)

            logger.info("using tokenizer: %s", path)
            return encode
        except ImportError as e:
            raise ImportError(
                "Install tokenizers package for production encode: pip install tokenizers"
            ) from e

    vs = max(2, min(int(vocab_size), 256))

    def encode_bytes(text: str) -> list[int]:
        return [b % vs for b in text.encode("utf-8", errors="ignore")]

    logger.warning(
        "No --tokenizer-json; using UTF-8 byte encode (cap %d). "
        "Pass tokenizer.json from prepare_data.py for production.",
        vs,
    )
    return encode_bytes


class PreferenceDataset(Dataset):
    def __init__(self, path: str, tokenizer_encode: EncodeFn, max_len: int = 256):
        path_p = Path(path)
        if not path_p.is_file():
            raise FileNotFoundError(f"preference file not found: {path}")
        self.rows = [normalize_pref_row(r) for r in _read_jsonl(path_p)]
        self.rows = [r for r in self.rows if r["chosen"] and r["rejected"]]
        self.encode = tokenizer_encode
        self.max_len = max_len
        if not self.rows:
            raise ValueError(f"No valid preference pairs in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        pc = self._pack(r["prompt"], r["chosen"])
        pr = self._pack(r["prompt"], r["rejected"])
        return {
            "chosen_x": pc["x"],
            "chosen_y": pc["y"],
            "rejected_x": pr["x"],
            "rejected_y": pr["y"],
        }

    def _pack(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        text = f"{prompt}\n{response}"
        ids = self.encode(text)[: self.max_len + 1]
        if len(ids) < 2:
            raise ValueError("encoded sequence too short; check tokenizer and data")
        t = torch.tensor(ids, dtype=torch.long)
        return {"x": t[:-1], "y": t[1:]}


class SFTTextDataset(Dataset):
    def __init__(self, path: str, tokenizer_encode: EncodeFn, max_len: int = 256):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"SFT file not found: {path}")
        rows: list[str] = []
        if p.suffix == ".jsonl":
            for r in _read_jsonl(p):
                t = normalize_sft_row(r)["text"].strip()
                if t:
                    rows.append(t)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
            rows = [c.strip() for c in text.split("\n\n") if c.strip()]
        self.rows = rows
        self.encode = tokenizer_encode
        self.max_len = max_len
        if not self.rows:
            raise ValueError(f"No SFT text in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.encode(self.rows[idx])[: self.max_len + 1]
        if len(ids) < 2:
            raise ValueError("encoded sequence too short; check tokenizer and data")
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t[:-1], "targets": t[1:]}


def collate_pad(batch: list[dict[str, torch.Tensor]], keys: list[str]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        seqs = [b[k] for b in batch]
        max_t = max(s.size(0) for s in seqs)
        padded = torch.zeros(len(seqs), max_t, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
        out[k] = padded
    return out


def preference_loader(path, encode, batch_size, max_len, num_workers=0):
    ds = PreferenceDataset(path, encode, max_len=max_len)
    keys = ["chosen_x", "chosen_y", "rejected_x", "rejected_y"]
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, keys), drop_last=True,
    )


def sft_loader(path, encode, batch_size, max_len, num_workers=0):
    ds = SFTTextDataset(path, encode, max_len=max_len)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, ["input_ids", "targets"]), drop_last=True,
    )
