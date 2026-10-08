"""Evaluation suite: held-out loss + generation smoke tests.

Changes vs the original:
  * Validation loss is computed ONLY on text passed as ``val_text`` (data the
    model never trained on). The old code split the *training* corpus 80/20,
    which made val_nll/perplexity meaningless.
  * If no held-out text is given, val metrics are reported as ``None`` with a
    warning instead of a flattering number.
  * NLL is pure cross-entropy (the old code returned the training loss, which
    includes auxiliary error/balance terms).
  * The identity check is optional and parameterised (no hardcoded product).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from neurofield.utils.split import split_blocks  # noqa: F401  (re-export)


@torch.no_grad()
def sequence_nll(model, ids: torch.Tensor, device: str = "cpu", max_tokens: int = 4096, chunk: int = 512) -> float:
    """Mean cross-entropy (nats/token) over a 1D token tensor, in independent chunks."""
    ids = ids[:max_tokens]
    if ids.numel() < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(0, ids.numel() - 1, chunk):
        seg = ids[i : i + chunk + 1]
        if seg.numel() < 2:
            break
        x = seg[:-1].unsqueeze(0).to(device)
        y = seg[1:].unsqueeze(0).to(device)
        logits = model(x).logits
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total += float(ce)
        count += y.numel()
    return total / max(1, count)


def generate_text(
    model,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    prompt: str,
    max_new: int = 48,
    temperature: float = 0.4,
    device: str = "cpu",
) -> str:
    model.eval()
    ids = encode(prompt) or [0]
    cur = torch.tensor([ids], device=device)
    state = None
    out_ids = list(ids)
    with torch.no_grad():
        for _ in range(max_new):
            o = model(cur, state=state)
            state = o.state
            logits = o.logits[0, -1]
            if temperature <= 1e-5:
                nid = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nid = int(torch.multinomial(probs, 1).item())
            out_ids.append(nid)
            cur = torch.tensor([[nid]], device=device)
    return decode(out_ids)


def run_eval_suite(
    model,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    text: Optional[str] = None,
    device: str = "cpu",
    prompts: list[str] | None = None,
    *,
    val_text: Optional[str] = None,
    identity_prompt: str = "who am I",
    identity_marker: Optional[str] = None,
) -> dict[str, Any]:
    """``text`` = training text (for train_nll only). ``val_text`` = held-out text."""
    model.eval()
    prompts = prompts or [identity_prompt, "namaste", "I am", "mera naam"]
    warnings: list[str] = []

    train_ids = encode(text) if text else []
    train_nll = (
        sequence_nll(model, torch.tensor(train_ids, dtype=torch.long), device) if len(train_ids) > 2 else None
    )

    val_nll = val_ppl = None
    val_ids: list[int] = encode(val_text) if val_text else []
    if len(val_ids) > 2:
        val_nll = sequence_nll(model, torch.tensor(val_ids, dtype=torch.long), device)
        val_ppl = math.exp(min(val_nll, 20)) if val_nll == val_nll else None
    else:
        warnings.append("no held-out val_text supplied: val_nll/val_perplexity NOT computed")

    gens: dict[str, str] = {}
    t0 = time.time()
    for p in prompts:
        gens[p] = generate_text(model, encode, decode, p, max_new=40, temperature=0.3, device=device)
    latency = time.time() - t0

    identity_hit: Optional[bool] = None
    if identity_marker:
        gen = gens.get(identity_prompt) or generate_text(
            model, encode, decode, identity_prompt, max_new=40, temperature=0.3, device=device
        )
        identity_hit = identity_marker.lower() in gen.lower()

    return {
        "train_nll": train_nll,
        "val_nll": val_nll,
        "val_perplexity": val_ppl,
        "val_tokens": len(val_ids),
        "identity_hit": identity_hit,
        "identity_marker": identity_marker,
        "generations": gens,
        "eval_latency_s": round(latency, 3),
        "num_train_tokens": len(train_ids),
        "warnings": warnings,
    }


def save_eval_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
