"""
Brain-inspired metacognition for NeuroField.

Human metacognition (roughly):
  MONITOR  — how wrong / uncertain am I? (prediction error, conflict, confidence)
  CONTROL  — allocate effort, tag memory for encoding, schedule consolidation

Mapped onto AESC control path (paper §3 / §6):
  surprise S          → prediction-error signal (anterior cingulate / locus coeruleus)
  router entropy      → decision conflict / uncertainty
  confidence          → feeling-of-knowing (inverse surprise, calibrated)
  arousal             → integrated urgency (surprise + conflict)
  k                   → computational depth ("think longer when unsure")
  gate_bias δ         → encoding priority (what gets written to fast memory)
  replay trigger      → systems consolidation (hippocampus → neocortex analogue)

Still rule-based with a few learnable scalars (not a full learned policy).
Learned policy is future work; this is the production control layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class ControlDecision:
    """Per-step metacognitive control outputs."""

    rel_S: Tensor          # (B,) relative surprise
    bar_S: Tensor          # scalar or 0-dim running baseline
    k: int                 # field steps this token
    gate_bias: Tensor      # (B,) δ added inside neuromodulator logit
    confidence: Tensor     # (B,) in (0, 1)
    arousal: Tensor        # (B,) non-negative urgency
    conflict: Tensor       # (B,) uncertainty from prior router entropy
    want_replay: bool      # consolidation request (trainer may honour)
    diagnostics: dict[str, float]


class Metacognition:
    """
    Monitor + control loop.

    State (session-local when track_global=False):
      bar_S, bar_ent, cum_surprise, tokens_seen, last_replay_token
    """

    def __init__(
        self,
        k_max: int = 4,
        momentum: float = 0.95,
        track_global: bool = True,
        # Replay: consolidate after enough cumulative surprise (brain: offline replay)
        replay_surprise_budget: float = 8.0,
        replay_min_tokens: int = 32,
        # Habituation floor so first tokens do not explode rel_S
        bar_floor: float = 0.05,
    ):
        self.k_max = max(1, int(k_max))
        self.momentum = float(momentum)
        self.track_global = bool(track_global)
        self.replay_surprise_budget = float(replay_surprise_budget)
        self.replay_min_tokens = int(replay_min_tokens)
        self._bar_floor = float(bar_floor)

        self.bar_S: Optional[Tensor] = None
        self.bar_ent: Optional[Tensor] = None
        self.cum_surprise: float = 0.0
        self.tokens_seen: int = 0
        self.last_replay_at: int = 0

        # Mild learnable calibration (optional optimizers can include these if registered)
        # Kept as plain floats so Metacognition stays outside nn.Module graph by default.
        self.conf_temperature: float = 1.0
        self.arousal_surprise_w: float = 0.7
        self.arousal_conflict_w: float = 0.3

    # ------------------------------------------------------------------ monitor
    def update_running_surprise(
        self,
        S: Tensor,
        bar_S: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Backward-compatible API used by older call sites."""
        decision = self.step(S=S, bar_S=bar_S, router_entropy=None)
        return decision.rel_S, decision.bar_S

    def choose_k(self, relative_surprise: Tensor) -> int:
        """Backward-compatible: k from relative surprise only."""
        return self._allocate_k(relative_surprise, conflict=None)

    def step(
        self,
        S: Tensor,
        bar_S: Optional[Tensor] = None,
        router_entropy: Optional[Tensor] = None,
        mem_norm: Optional[Tensor] = None,
    ) -> ControlDecision:
        """
        Full monitor→control step.

        Args:
          S: (B,) per-sequence surprise (detached preferred).
          bar_S: optional session baseline (else module / global).
          router_entropy: (B,) or scalar from *previous* step (conflict proxy).
          mem_norm: optional fast-memory norm (load signal).
        """
        S = S.detach()
        device = S.device
        B = S.shape[0]

        mean_S = S.mean()
        if bar_S is None and self.track_global:
            bar_S = self.bar_S
        if bar_S is None:
            new_bar = mean_S.clamp(min=self._bar_floor)
        else:
            new_bar = self.momentum * bar_S + (1.0 - self.momentum) * mean_S
            new_bar = new_bar.clamp(min=self._bar_floor)
        if self.track_global:
            self.bar_S = new_bar

        rel_S = S / (new_bar + 1e-8)

        # Conflict from prior router entropy (high entropy = unsure which skill)
        if router_entropy is None:
            conflict = torch.zeros(B, device=device)
            ent_mean = torch.tensor(0.0, device=device)
        else:
            ent = router_entropy.detach()
            if ent.ndim == 0:
                conflict = ent.expand(B)
                ent_mean = ent
            else:
                conflict = ent.reshape(-1)
                if conflict.numel() != B:
                    conflict = conflict.mean().expand(B)
                ent_mean = conflict.mean()
            # running entropy baseline (session-local unless track_global)
            if self.track_global:
                if self.bar_ent is None:
                    self.bar_ent = ent_mean
                else:
                    self.bar_ent = self.momentum * self.bar_ent + (1.0 - self.momentum) * ent_mean
                bar_e = self.bar_ent
            else:
                bar_e = ent_mean.clamp(min=1e-4)
            conflict = conflict / (bar_e + 1e-8)

        # Confidence ≈ feeling of knowing: high when surprise low vs baseline
        # sigmoid so (0,1); temperature softens calibration
        conf = torch.sigmoid(-self.conf_temperature * (rel_S - 1.0))

        # Arousal / urgency: blend surprise and conflict (LC–NE style)
        arousal = (
            self.arousal_surprise_w * rel_S
            + self.arousal_conflict_w * conflict
        ).clamp(min=0.0)

        # Control: effort (k) and encoding tag (gate_bias)
        k = self._allocate_k(rel_S, conflict)
        gate_bias = self._encoding_bias(rel_S, conf, conflict)

        # Consolidation request (offline replay analogue)
        if self.track_global:
            self.tokens_seen += 1
            self.cum_surprise += float(mean_S.item())
            want_replay = self._should_consolidate()
        else:
            want_replay = False

        # Optional load: if memory already huge, slightly lower encoding bias
        if mem_norm is not None:
            mn = float(mem_norm.detach().mean().item()) if mem_norm.numel() else 0.0
            if mn > 30.0:
                gate_bias = gate_bias - 0.25

        diag = {
            "bar_S": float(new_bar.item()) if new_bar.ndim == 0 else float(new_bar.mean().item()),
            "rel_S_mean": float(rel_S.mean().item()),
            "confidence_mean": float(conf.mean().item()),
            "arousal_mean": float(arousal.mean().item()),
            "conflict_mean": float(conflict.mean().item()),
            "k": float(k),
            "want_replay": 1.0 if want_replay else 0.0,
            "cum_surprise": self.cum_surprise,
        }

        return ControlDecision(
            rel_S=rel_S,
            bar_S=new_bar,
            k=k,
            gate_bias=gate_bias,
            confidence=conf,
            arousal=arousal,
            conflict=conflict,
            want_replay=want_replay,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------ control
    def _allocate_k(self, rel_S: Tensor, conflict: Optional[Tensor]) -> int:
        """
        More computation when surprised or conflicted — like re-checking under uncertainty.
        Uses batch median for stability.
        """
        s = rel_S.median().item()
        c = 0.0 if conflict is None else float(conflict.median().item())
        # surprise channel (paper formula) + conflict boost
        frac_s = max(0.0, min(1.0, (s - 0.5) / 1.5))
        frac_c = max(0.0, min(1.0, (c - 0.5) / 1.5))
        frac = max(0.0, min(1.0, 0.75 * frac_s + 0.25 * frac_c))
        k = 1 + int(round((self.k_max - 1) * frac))
        return max(1, min(self.k_max, k))

    def _encoding_bias(self, rel_S: Tensor, conf: Tensor, conflict: Tensor) -> Tensor:
        """
        Memory-tagging bias δ (paper discrete ±0.5, here smooth + conflict).

        High surprise + low confidence → prioritize write (novel / errorful events).
        High confidence + low surprise → down-weight write (already known).
        """
        # Paper thresholds as soft anchors
        delta = torch.zeros_like(rel_S)
        delta = torch.where(rel_S > 1.5, torch.full_like(delta, 0.5), delta)
        delta = torch.where(rel_S < 0.5, torch.full_like(delta, -0.5), delta)
        # Continuous refinement: low confidence boosts encoding
        delta = delta + 0.25 * (1.0 - conf) + 0.15 * conflict.clamp(max=2.0)
        return delta.clamp(-1.0, 1.0)

    def _should_consolidate(self) -> bool:
        if self.tokens_seen - self.last_replay_at < self.replay_min_tokens:
            return False
        if self.cum_surprise >= self.replay_surprise_budget:
            return True
        return False

    def acknowledge_replay(self) -> None:
        """Trainer calls this after slow-memory replay so budget resets."""
        self.last_replay_at = self.tokens_seen
        self.cum_surprise = 0.0

    def should_replay(self, step: int, every: int) -> bool:
        """Legacy periodic schedule OR surprise-budget consolidation."""
        periodic = every > 0 and step > 0 and step % every == 0
        return periodic or self._should_consolidate()

    def reset(self) -> None:
        self.bar_S = None
        self.bar_ent = None
        self.cum_surprise = 0.0
        self.tokens_seen = 0
        self.last_replay_at = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "bar_S": None if self.bar_S is None else float(self.bar_S.detach().cpu().item()),
            "bar_ent": None if self.bar_ent is None else float(self.bar_ent.detach().cpu().item()),
            "cum_surprise": self.cum_surprise,
            "tokens_seen": self.tokens_seen,
            "last_replay_at": self.last_replay_at,
        }
