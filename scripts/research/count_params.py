#!/usr/bin/env python3
"""Exact NeuroField parameter count from a config — no torch needed.

    python scripts/count_params.py configs/aeris_1b.yaml
    python scripts/count_params.py --preset aeris_1b
    python scripts/count_params.py --checkpoint docs/neurofield_india   # reads safetensors header

The formula is validated by tests against real checkpoint headers.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import yaml


def count(V, D, dk, dv, n_skills, window, tie=True) -> dict[str, int]:
    parts = {
        "embed": V * D,
        "head(untied)": 0 if tie else V * D,
        "dendrite": window * D + 2 * D,
        "field": 3 * D + dv * D + (3 * D * D + D) + 2 * D,
        "predictor": 2 * (D * D + D),
        "neuromodulator": 3,
        "router": D * n_skills,
        "skills": n_skills * (D * 2 * D + 2 * D + 2 * D * D + D),
        "fast_memory": 2 * D * dk + D * dv + dk,
        "slow_memory": D * dk + dk * dv,
        "P_f+P_s": 2 * dv * D,
        "mem_scale+out_norm": 1 + 2 * D,
    }
    parts["TOTAL"] = sum(parts.values())
    return parts


def from_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: v["shape"] for k, v in h.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--preset")
    ap.add_argument("--checkpoint")
    a = ap.parse_args()
    if a.checkpoint:
        sh = from_header(Path(a.checkpoint) / "model.safetensors")
        V, D = sh["embed.weight"]
        m = dict(vocab_size=V, d_model=D, d_k=sh["fast_mem.W_k.weight"][0], d_v=sh["fast_mem.W_v.weight"][0],
                 n_skills=sh["router.router.weight"][0], dendrite_window=sh["dendrite.coeffs"][0],
                 tie_embeddings=True)
    elif a.preset:
        raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/presets.yaml").read_text())
        m = raw[a.preset]
    elif a.config:
        m = yaml.safe_load(Path(a.config).read_text())["model"]
    else:
        sys.exit("give a config path, --preset or --checkpoint")
    p = count(m["vocab_size"], m["d_model"], m["d_k"], m["d_v"], m["n_skills"],
              m["dendrite_window"], m.get("tie_embeddings", True))
    for k, v in p.items():
        print(f"{k:>20}: {v:>16,}")
    print(f"{'':>20}  = {p['TOTAL'] / 1e9:.4f} B parameters")


if __name__ == "__main__":
    main()
