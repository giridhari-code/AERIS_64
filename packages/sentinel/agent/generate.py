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
    Load SENTINEL checkpoint once; return prompt -> text.

    Uses ``load_model_from_checkpoint`` (same path as the server).
    On failure, returns a function that yields "" and sets ``fn.load_error``.
    """

    def _blank(reason: str) -> Callable[[str], str]:
        def _fn(_p: str) -> str:
            return ""

        _fn.load_error = reason  # type: ignore[attr-defined]
        return _fn

    path = Path(checkpoint)
    if not path.exists():
        return _blank(f"missing path: {path}")

    try:
        from sentinel.checkpoint.resolve import load_model_from_checkpoint
    except Exception as e:
        return _blank(f"import resolve: {e}")

    try:
        model, tok, _cfg, _meta = load_model_from_checkpoint(
            str(path),
            config_path=None,
            device=device,
            strict=False,
        )
        model.eval()
        encode = tok.encode
        decode = tok.decode
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
            # clamp ids into vocab if possible
            vs = int(getattr(tok, "vocab_size", 0) or 0)
            if vs > 0:
                ids = [min(max(0, int(i)), vs - 1) for i in ids]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            state = None
            out_ids: list[int] = []
            for _ in range(max_new_tokens):
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
        except Exception as e:
            generate.last_error = f"{type(e).__name__}: {e}"  # type: ignore[attr-defined]
            return ""

    generate.load_error = None  # type: ignore[attr-defined]
    generate.last_error = None  # type: ignore[attr-defined]
    return generate
