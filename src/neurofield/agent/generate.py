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

    On failure the returned function still runs but yields "" and stores
    reason on ``fn.load_error`` for diagnostics.
    """
    load_error: Optional[str] = None

    def _blank(reason: str) -> Callable[[str], str]:
        nonlocal load_error
        load_error = reason

        def _fn(_p: str) -> str:
            return ""

        _fn.load_error = reason  # type: ignore[attr-defined]
        return _fn

    try:
        from neurofield.checkpoint.io import load_checkpoint
        from neurofield.model import NeuroField
    except Exception as e:
        return _blank(f"import: {e}")

    path = Path(checkpoint)
    if not path.exists():
        return _blank(f"missing path: {path}")

    try:
        bundle = load_checkpoint(str(path), map_location=device)
        model = None
        encode = decode = None

        if isinstance(bundle, dict):
            model = bundle.get("model")
            encode = bundle.get("encode")
            decode = bundle.get("decode")
            tok = bundle.get("tokenizer")
            if tok is not None and (encode is None or decode is None):
                encode, decode = tok.encode, tok.decode
            # some loaders nest
            if model is None and "state_dict" in bundle:
                from neurofield.config import ModelConfig, SafetyConfig
                # cannot rebuild without config — fail clear
                return _blank("checkpoint dict has state_dict but no model object")
        else:
            model = bundle

        if model is not None and not isinstance(model, NeuroField):
            model = getattr(model, "model", model)

        if model is None:
            return _blank("no model in checkpoint bundle")

        model = model.to(device)
        model.eval()

        if encode is None or decode is None:
            # try vocab files beside ckpt
            return _blank("no tokenizer encode/decode in checkpoint (train/save tokenizer with ckpt)")

    except Exception as e:
        return _blank(f"load: {type(e).__name__}: {e}")

    @torch.no_grad()
    def generate(prompt: str) -> str:
        try:
            ids = encode(prompt)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids:
                return ""
            x = torch.tensor([ids], dtype=torch.long, device=device)
            state = None
            out_ids = []
            for i in range(max_new_tokens):
                inp = x if state is None else x[:, -1:]
                out = model(inp, state=state)
                state = out.state
                logits = out.logits[:, -1, :]
                if temperature and temperature > 1e-5:
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                    nid = int(torch.multinomial(probs, 1).item())
                else:
                    nid = int(logits.argmax(-1).item())
                out_ids.append(nid)
                x = torch.tensor([[nid]], dtype=torch.long, device=device)
                if len(out_ids) > 8 and len(set(out_ids[-6:])) == 1:
                    break
            text = decode(out_ids)
            return text if isinstance(text, str) else str(text)
        except Exception:
            return ""

    generate.load_error = None  # type: ignore[attr-defined]
    return generate
