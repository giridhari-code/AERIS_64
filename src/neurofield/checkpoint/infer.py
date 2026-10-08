"""Infer architecture dimensions from a state_dict (shapes are ground truth).

Fixes the case where ``config.json`` is empty/missing/wrong: the weights
themselves say what d_model, vocab, n_skills ... must be. Pure python (accepts
tensors or plain shape tuples) so it can be tested without torch.
"""

from __future__ import annotations

from typing import Any, Mapping


def _shape(v: Any) -> tuple[int, ...]:
    shp = getattr(v, "shape", v)
    return tuple(int(x) for x in shp)


def infer_model_dims(state: Mapping[str, Any]) -> dict[str, int]:
    """Return {vocab_size,d_model,d_k,d_v,n_skills,dendrite_window} found in state."""
    out: dict[str, int] = {}
    if "embed.weight" in state:
        v, d = _shape(state["embed.weight"])
        out["vocab_size"], out["d_model"] = v, d
    if "fast_mem.W_k.weight" in state:
        out["d_k"] = _shape(state["fast_mem.W_k.weight"])[0]
    if "fast_mem.W_v.weight" in state:
        out["d_v"] = _shape(state["fast_mem.W_v.weight"])[0]
    if "router.router.weight" in state:
        out["n_skills"] = _shape(state["router.router.weight"])[0]
    if "dendrite.coeffs" in state:
        out["dendrite_window"] = _shape(state["dendrite.coeffs"])[0]
    return out


def count_params_from_shapes(state: Mapping[str, Any], tied_head: bool = True) -> int:
    """Unique parameter count of a saved state (tied head counted once)."""
    total = 0
    for k, v in state.items():
        if tied_head and k == "head.weight":
            continue
        n = 1
        for x in _shape(v):
            n *= x
        total += n
    return total


# Old checkpoints used a different tensor name. The (d_k, d_v) matrix is the
# same object; leaving it unmapped made servers load with a RANDOM slow memory
# (strict=False hid it).
LEGACY_KEY_MAP = {"slow_mem.W_s": "slow_mem.M_s"}


def remap_legacy_keys(state: dict) -> tuple[dict, list[str]]:
    """Return (new_state, renamed_keys). Only renames when the new name is absent."""
    out = dict(state)
    renamed: list[str] = []
    for old, new in LEGACY_KEY_MAP.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
            renamed.append(f"{old} -> {new}")
    return out, renamed
