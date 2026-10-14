"""Optional checkpoint-backed generate_fn for SI_v0 agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch


def make_generate_fn(
    checkpoint: str,
    device: str = "cpu",
    max_new_tokens: int = 64,
    temperature: float = 0.4,
) -> Callable[[str], str]:
    """
    Load NeuroField checkpoint once; return prompt -> text.

    If load fails, returns a function that yields empty string (agent falls back to facts).
    """
    try:
        from neurofield.checkpoint.io import load_checkpoint
        from neurofield.model import NeuroField
    except Exception:
        def _empty(_p: str) -> str:
            return ""
        return _empty

    path = Path(checkpoint)
    if not path.exists():
        def _missing(_p: str) -> str:
            return ""
        return _missing

    try:
        bundle = load_checkpoint(str(path), map_location=device)
        model: NeuroField = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        if not isinstance(model, NeuroField):
            # common layout: object with .model
            model = getattr(bundle, "model", model)
        model = model.to(device)
        model.eval()
        encode = bundle.get("encode") if isinstance(bundle, dict) else None
        decode = bundle.get("decode") if isinstance(bundle, dict) else None
        if encode is None or decode is None:
            tok = bundle.get("tokenizer") if isinstance(bundle, dict) else None
            if tok is not None:
                encode, decode = tok.encode, tok.decode
            else:
                # char fallback
                def encode(s: str):
                    return [ord(c) % 100 for c in s][:256]

                def decode(ids):
                    return "".join(chr(int(i) % 128) for i in ids)
    except Exception:
        def _fail(_p: str) -> str:
            return ""
        return _fail

    @torch.no_grad()
    def generate(prompt: str) -> str:
        ids = encode(prompt)
        if not ids:
            return ""
        x = torch.tensor([ids], dtype=torch.long, device=device)
        state = None
        out_ids = []
        for _ in range(max_new_tokens):
            out = model(x[:, -1:] if state is not None else x, state=state)
            state = out.state
            logits = out.logits[:, -1, :]
            if temperature and temperature > 1e-5:
                probs = torch.softmax(logits / temperature, dim=-1)
                nid = int(torch.multinomial(probs, 1).item())
            else:
                nid = int(logits.argmax(-1).item())
            out_ids.append(nid)
            x = torch.tensor([[nid]], dtype=torch.long, device=device)
            # soft stop on repeated newline-ish
            if len(out_ids) > 8 and len(set(out_ids[-6:])) == 1:
                break
        try:
            return decode(out_ids)
        except Exception:
            return ""

    return generate
