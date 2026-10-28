#!/usr/bin/env python3
"""Show what is "on the model's mind": J-lens readout + active workspace atoms.

    python scripts/research/workspace_probe.py --text "the cat sat on the"
    python scripts/research/workspace_probe.py --checkpoint runs/aeris_109m/best --text "..."

Per position it prints the top tokens of  softmax(W_U . norm(J h))  (J = I + B A, so an
untrained workspace is exactly the logit lens) and the strongest active atoms.
Works on any checkpoint: missing ``workspace.*`` weights just stay at their init.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from sentinel.checkpoint import load_checkpoint  # noqa: E402
from sentinel.config import ModelConfig  # noqa: E402
from sentinel.model import SENTINEL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--checkpoint", default=None, help="folder with model.safetensors")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--k-depth", type=int, default=3)
    a = ap.parse_args()

    fields = ModelConfig.__dataclass_fields__
    stoi, itos, state = {}, {}, None
    if a.checkpoint:
        ck = load_checkpoint(a.checkpoint)
        raw = {k: v for k, v in (ck["config"] or {}).items() if k in fields}
        stoi, itos, state = ck["stoi"], ck["itos"], ck["model"]
    else:
        raw = dict(d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=3, vocab_size=256, max_seq_len=256)
        print("[probe] no checkpoint: random init (readout is only a wiring demo)", file=sys.stderr)
    raw.update(workspace_mode="learned")
    cfg = ModelConfig(**raw)
    m = SENTINEL(cfg).eval()
    if state is not None:
        res = m.load_state_dict(state, strict=False)
        miss = [k for k in res.missing_keys if k.startswith("workspace.")]
        if miss:
            print(f"[probe] {len(miss)} workspace.* tensors not in checkpoint -> init (J = I, i.e. logit lens)", file=sys.stderr)

    ids = [stoi.get(c, 0) for c in a.text] if stoi else list(a.text.encode("utf-8"))
    ids_t = torch.tensor([ids], dtype=torch.long)
    dec = (lambda i: itos.get(int(i), "?")) if itos else (lambda i: bytes([int(i) % 256]).decode("latin-1"))

    seen = {}
    hook = m.workspace.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
    with torch.no_grad():
        m(ids_t, k_depth=a.k_depth)
    hook.remove()

    h, code = seen["h"], m._last_ws_code  # final refined field state / final code, (T, D) / (T, atoms)
    tok, prob = m.workspace.read_tokens(h, m.head.weight, topn=a.topn)
    print(f"{'pos':>3} {'char':>4}  {'J-lens top tokens':<42} strongest atoms")
    for t in range(h.size(0)):
        words = " ".join(f"{dec(i)!r}:{p:.2f}" for i, p in zip(tok[t].tolist(), prob[t].tolist()))
        top = code[t].topk(min(3, code.size(-1)))
        atoms = " ".join(f"#{i}({v:.2f})" for v, i in zip(top.values.tolist(), top.indices.tolist()) if v > 0)
        print(f"{t:>3} {dec(ids[t])!r:>4}  {words:<42} {atoms}")

    freq = Counter(i for row in (code > 0).nonzero().tolist() for i in [row[1]])
    aw = m.workspace.atom_words(m.head.weight, topn=4)
    print("\nmost-used atoms -> tokens they push toward:")
    for i, c in freq.most_common(5):
        print(f"  atom #{i:<3} used {c}x -> {[dec(x) for x in aw[i].tolist()]}")


if __name__ == "__main__":
    main()
