"""Inference cache for SENTINEL — recurrent analogue of Transformer KV cache.

Classic Transformer KV cache grows as O(layers × T × heads × dim) and stores
past keys/values so decode is O(1) new-token attention.

SENTINEL has no multi-layer attention stack. Its fixed-size fast M + slow ring
already bound memory. This module still provides a **KV-cache-style API** for
inference speedup:

  * Carry full session state across decode steps (no re-prefill)
  * Cache slow-ring slot keys (avoid recomputing fingerprints every token)
  * Cache last dendrite / field vectors for cheap single-token steps
  * Optional static “prompt cache”: freeze slow ring after prefill
  * Health-aware (ties into self-heal)

Usage
-----
    cache = InferenceCache.from_model(model, batch=1, device="cuda")
    cache.prefill(model, prompt_ids)          # one parallel pass
    for _ in range(max_new):
        tok = cache.decode_step(model, next_id)  # T=1, state carried
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from .gk_cache import GKCache, get_gk_cache


@dataclass
class InferenceCache:
    """Session-local inference state (SENTINEL's KV-cache equivalent)."""

    state: dict[str, Any]
    device: torch.device
    batch: int = 1

    # Cached slow-ring keys: (B, S, d_k) — invalidated on ring write
    slot_keys: Optional[Tensor] = None
    slot_keys_valid: bool = False

    # Prefill marker: after prefill, prompt is "cached" in slow/fast state
    prefilled: bool = False
    tokens_generated: int = 0
    prompt_len: int = 0

    # Stats
    prefill_tokens: int = 0
    decode_steps: int = 0

    # General Knowledge cache (shared or session-local)
    gk: Optional[GKCache] = None

    @classmethod
    def from_model(
        cls,
        model: Any,
        batch: int = 1,
        device: Optional[torch.device] = None,
        slow: Optional[dict] = None,
    ) -> "InferenceCache":
        dev = device or next(model.parameters()).device
        state = model.new_state(batch, dev, slow=slow)
        return cls(state=state, device=dev, batch=batch, gk=get_gk_cache())

    def invalidate_slot_keys(self) -> None:
        self.slot_keys = None
        self.slot_keys_valid = False
        if isinstance(self.state, dict):
            self.state.pop("slow_slot_keys", None)

    def update_slot_keys(self, model: Any) -> None:
        """Cache content-address keys for current slow ring (speed up repeated reads)."""
        slots = self.state.get("slow_slots") if isinstance(self.state, dict) else None
        if slots is None or not hasattr(model, "slow_mem"):
            self.invalidate_slot_keys()
            return
        with torch.inference_mode():
            keys = model.slow_mem._slot_keys(slots).detach()
        self.slot_keys = keys
        self.slot_keys_valid = True
        if isinstance(self.state, dict):
            self.state["slow_slot_keys"] = keys

    def heal(self, model: Any) -> None:
        if hasattr(model, "heal_session_state"):
            self.state = model.heal_session_state(self.state) or self.state
        self.invalidate_slot_keys()

    @torch.inference_mode()
    def prefill(self, model: Any, input_ids: Tensor) -> Tensor:
        """Process prompt in one parallel pass; state becomes the 'cache'."""
        model.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        # collect_stats=False: metrics are unused here and cost many host syncs
        out = model(input_ids, state=self.state, collect_stats=False)
        self.state = out.state or self.state
        self.prefilled = True
        self.prompt_len = int(input_ids.size(1))
        self.prefill_tokens += self.prompt_len
        self.update_slot_keys(model)
        return out.logits

    @torch.inference_mode()
    def decode_step(
        self,
        model: Any,
        token_id: Tensor,
        *,
        heal_every: int = 64,
    ) -> Tensor:
        """Single-token decode using carried state (O(1) memory growth)."""
        model.eval()
        if token_id.dim() == 0:
            token_id = token_id.view(1, 1)
        elif token_id.dim() == 1:
            token_id = token_id.view(-1, 1)
        token_id = token_id.to(self.device)

        if heal_every > 0 and self.decode_steps > 0 and self.decode_steps % heal_every == 0:
            self.heal(model)

        if self.slot_keys_valid and self.slot_keys is not None and isinstance(self.state, dict):
            self.state["slow_slot_keys"] = self.slot_keys

        out = model(token_id, state=self.state, collect_stats=False)
        self.state = out.state or self.state
        self.decode_steps += 1
        self.tokens_generated += 1
        # Ring may have advanced — refresh keys lazily next read if needed
        if self.state.get("seg_pos", 0) == 0:
            self.invalidate_slot_keys()
        return out.logits

    @torch.inference_mode()
    def generate(
        self,
        model: Any,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_id: Optional[int] = None,
        heal_every: int = 64,
    ) -> Tensor:
        """Prefill + autoregressive decode; returns full sequence (prompt + new)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        logits = self.prefill(model, input_ids)
        generated = [input_ids]
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :]  # (B, V)
            if temperature <= 0:
                next_id = step_logits.argmax(dim=-1)
            else:
                logits_t = step_logits / max(temperature, 1e-6)
                if top_k > 0:
                    v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
                    logits_t = logits_t.masked_fill(logits_t < v[:, -1:], float("-inf"))
                probs = torch.softmax(logits_t, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_id = next_id.view(-1, 1)
            if eos_id is not None:
                # rows that already emitted EOS keep emitting EOS (no garbage after the end of an answer)
                next_id = torch.where(finished.view(-1, 1), torch.full_like(next_id, eos_id), next_id)
                finished = finished | (next_id.view(-1) == eos_id)
            generated.append(next_id)
            if eos_id is not None and bool(finished.all()):
                break
            logits = self.decode_step(model, next_id, heal_every=heal_every)

        return torch.cat(generated, dim=1)


    def gk_lookup(self, query: str, top_k: int = 4) -> list:
        if self.gk is None:
            return []
        return self.gk.lookup(query, top_k=top_k)

    def gk_put(self, text: str, source: str = "session", score: float = 1.0) -> str:
        if self.gk is None:
            self.gk = get_gk_cache()
        return self.gk.put(text, source=source, score=score)

    def gk_context(self, query: str, top_k: int = 3) -> str:
        if self.gk is None:
            return ""
        return self.gk.format_context(query, top_k=top_k)

    def stats(self) -> dict[str, float]:
        return {
            "prefill_tokens": float(self.prefill_tokens),
            "decode_steps": float(self.decode_steps),
            "tokens_generated": float(self.tokens_generated),
            "prompt_len": float(self.prompt_len),
            "slot_keys_cached": 1.0 if self.slot_keys_valid else 0.0,
            "gk_size": float(len(self.gk) if self.gk else 0),
            "gk_hit_rate": float(self.gk.summary()["hit_rate"] if self.gk else 0.0),
        }

    def reset(self, model: Any, slow: Optional[dict] = None) -> None:
        self.state = model.new_state(self.batch, self.device, slow=slow)
        self.invalidate_slot_keys()
        self.prefilled = False
        self.tokens_generated = 0
        self.prompt_len = 0
