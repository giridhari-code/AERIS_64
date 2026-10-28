"""
Brain-inspired metacognition for SENTINEL + Global Workspace (J-space).

Human metacognition (roughly):
  MONITOR  — how wrong / uncertain am I? (prediction error, conflict, confidence)
  CONTROL  — allocate effort, tag memory for encoding, schedule consolidation

Mapped onto AESC control path (paper §3 / §6):
  surprise S          → prediction-error signal (anterior cingulate / locus coeruleus)
  router entropy      → decision conflict / uncertainty
  confidence          → feeling-of-knowing (inverse surprise, calibrated)
  arousal             → integrated urgency (surprise + conflict)
  k                   → computational depth ("think longer when unsure")
  process_mode        → dual-process under control: system1 (fast) | system2 (deliberate)
  gate_bias δ         → encoding priority (what gets written to fast memory)
  replay trigger      → systems consolidation (hippocampus → neocortex analogue)

--------------------------------------------------------------------
J-SPACE  (Global Workspace analogue, inspired by Anthropic 2026)
--------------------------------------------------------------------
Anthropic ("A global workspace in language models", Gurnee et al.):
  A small set of special internal patterns (the *J-space*) that are:
    - reportable (model can say what is "on its mind")
    - controllable / steerable
    - used for multi-step internal reasoning
    - broadcast to the rest of the network
    - distinct from routine fluent processing

In SENTINEL we implement an architectural, capacity-limited workspace
inside Metacognition that:

  * maintains a sparse set of active concept slots (default 16)
  * is updated from surprise, conflict and (optional) field state h
  * strengthens under System-2 / high arousal
  * produces a broadcast vector that can bias field / router / gate
  * exposes read / inject / clear APIs for interpretability & steering

This is a functional GWT-style mechanism, not a fitted Jacobian lens
(which requires a trained unembedding + corpus Jacobians). It gives the
same control affordances: limited capacity, reportability, steerability,
and causal influence on deliberate processing.

Still rule-based with a few learnable scalars (not a full learned policy).
Learned policy / true J-lens fitting is future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from .consciousness import ConsciousnessSimulator, ConsciousnessState


@dataclass
class JSpaceState:
    """Snapshot of the global workspace (J-space) at one step."""

    # Broadcast summary that other modules can consume (B, d_ws) or (d_ws,)
    broadcast: Tensor
    # Soft occupancy of the concept slots (B, n_slots) or (n_slots,)
    occupancy: Tensor
    # How full the workspace is (0..1)
    capacity_used: float
    # Approximate number of meaningfully active slots
    n_active: int
    # Whether contents are considered reportable / verbalizable
    reportable: bool = True
    # Optional free-text / token labels if a dictionary is attached later
    labels: Optional[list[str]] = None


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
    process_mode: str = "system1"  # system1 (fast) | system2 (deliberate)
    deliberate: bool = False       # True when system2
    # --- J-space / Global Workspace ---
    jspace: Optional[JSpaceState] = None
    jspace_broadcast: Optional[Tensor] = None  # convenience alias for broadcast vector
    # --- Consciousness access simulator ---
    consciousness: Optional["ConsciousnessState"] = None


class Metacognition:
    """
    Monitor + control loop + capacity-limited Global Workspace (J-space).

    State (session-local when track_global=False):
      bar_S, bar_ent, cum_surprise, tokens_seen, last_replay_token
      + J-space slot state (occupancy, workspace vectors)
    """

    def __init__(
        self,
        k_max: int = 5,
        momentum: float = 0.95,
        track_global: bool = True,
        # Replay: consolidate after enough cumulative surprise (brain: offline replay)
        replay_surprise_budget: float = 8.0,
        replay_min_tokens: int = 32,
        # Habituation floor so first tokens do not explode rel_S
        bar_floor: float = 0.05,
        # --- J-space / Global Workspace hyper-parameters ---
        jspace_slots: int = 16,          # limited capacity (Anthropic ~20-25)
        jspace_dim: int = 64,            # internal concept dimension
        jspace_decay: float = 0.85,      # slot decay per token (forgetting)
        jspace_write_threshold: float = 0.35,  # min arousal/conflict to write new
        jspace_sparsity: float = 0.25,   # soft top-k fraction kept active
        # True = a learned workspace (modules/workspace.py) owns the broadcast. The rule-based
        # slot bank is then skipped (no module-level state, no host syncs) and, in eval, the
        # consciousness baseline is not carried between calls -> same input => same logits.
        external_workspace: bool = False,
    ):
        self.k_max = max(1, int(k_max))
        self._last_process_mode = "system1"  # hysteresis for dual-process
        self.momentum = float(momentum)
        self.track_global = bool(track_global)
        self.training = True  # mirrored from the owning model's train()/eval() on every forward
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

        # ---------------- J-space state ----------------
        self.jspace_slots = max(4, int(jspace_slots))
        self.jspace_dim = max(8, int(jspace_dim))
        self.jspace_decay = float(jspace_decay)
        self.jspace_write_threshold = float(jspace_write_threshold)
        self.jspace_sparsity = float(jspace_sparsity)
        self.external_workspace = bool(external_workspace)

        # Slot bank: (n_slots, d_ws) concept vectors (session-persistent when track_global)
        self._js_slots: Optional[Tensor] = None          # (n_slots, d)
        self._js_occupancy: Optional[Tensor] = None      # (n_slots,) soft activation
        self._js_broadcast: Optional[Tensor] = None      # last broadcast (d,)

        # Consciousness-access simulator (GWT + φ-proxy)
        self.consciousness = ConsciousnessSimulator()

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
        n_tokens: int = 1,
        h: Optional[Tensor] = None,          # optional field state for richer workspace write
        session_jspace: Optional[dict] = None,  # rule-mode slot bank from Sequence.state
    ) -> ControlDecision:
        """
        Full monitor→control step + J-space update.

        Args:
            S: surprise / prediction error (B,) or scalar
            bar_S: optional external baseline
            router_entropy: optional conflict signal
            mem_norm: optional fast-memory norm (load shedding)
            n_tokens: how many tokens this call covers (train parallel)
            h: optional current neural-field state (B, d) or (d,) to seed workspace
            session_jspace: when not using global baselines, restore rule-mode slots
                from this pack (and caller should export_jspace_state() back to session)
        """
        # ---- original surprise / conflict / confidence logic (unchanged core) ----
        if S.ndim == 0:
            S = S.unsqueeze(0)
        B = S.shape[0]
        device = S.device
        mean_S = S.mean()

        use_global = self._use_global
        if bar_S is not None:
            new_bar = bar_S if bar_S.ndim == 0 else bar_S.mean()
        elif use_global:
            if self.bar_S is None:
                self.bar_S = mean_S.detach().clamp(min=self._bar_floor)
            else:
                self.bar_S = self.momentum * self.bar_S + (1.0 - self.momentum) * mean_S.detach()
            new_bar = self.bar_S
        else:
            new_bar = mean_S.detach().clamp(min=self._bar_floor)

        rel_S = (S / (new_bar + 1e-8)).clamp(min=0.0)

        # Conflict from router entropy
        conflict = torch.zeros(B, device=device)
        if router_entropy is not None:
            ent = router_entropy
            if ent.ndim == 0:
                conflict = ent.expand(B)
                ent_mean = ent
            else:
                conflict = ent.reshape(-1)
                if conflict.numel() != B:
                    conflict = conflict.mean().expand(B)
                ent_mean = conflict.mean()
            if use_global:
                if self.bar_ent is None:
                    self.bar_ent = ent_mean.detach()
                else:
                    self.bar_ent = self.momentum * self.bar_ent + (1.0 - self.momentum) * ent_mean.detach()
                bar_e = self.bar_ent
            else:
                bar_e = ent_mean.clamp(min=1e-4)
            conflict = conflict / (bar_e + 1e-8)

        conf = torch.sigmoid(-self.conf_temperature * (rel_S - 1.0))
        arousal = (
            self.arousal_surprise_w * rel_S
            + self.arousal_conflict_w * conflict
        ).clamp(min=0.0)

        process_mode, k = self.dual_process(rel_S, conflict, conf)
        gate_bias = self._encoding_bias(rel_S, conf, conflict)

        if use_global:
            self.tokens_seen += n_tokens
            self.cum_surprise += float(mean_S.item()) * n_tokens
            want_replay = self._should_consolidate()
        else:
            want_replay = False

        if mem_norm is not None:
            mn = float(mem_norm.detach().mean().item()) if mem_norm.numel() else 0.0
            if mn > 30.0:
                gate_bias = gate_bias - 0.25

        # ---- J-space update (Global Workspace) ----
        # Rule-mode slots live on the module for the duration of this call only when
        # _use_global is False: restore from session pack first so users cannot leak.
        if not use_global and not self.external_workspace:
            self.import_jspace_state(session_jspace, device=device)
        if self.external_workspace:
            js_state = self._neutral_jspace(device, B)
        else:
            js_state = self._update_jspace(
                rel_S=rel_S,
                conflict=conflict,
                arousal=arousal,
                conf=conf,
                process_mode=process_mode,
                h=h,
                device=device,
                B=B,
            )

        # When workspace is busy / high capacity, bias toward deeper processing
        if js_state.capacity_used > 0.55 and process_mode == "system1":
            process_mode = "system2"
            self._last_process_mode = "system2"
            k = max(k, min(self.k_max, 2 + int(js_state.capacity_used * (self.k_max - 1))))

        _full_diag = bool(getattr(self, "training", False)) or (self.tokens_seen % 8 == 0)
        if _full_diag:
            diag = {
                "bar_S": float(new_bar.detach().mean().item()),
                "rel_S_mean": float(rel_S.detach().mean().item()),
                "confidence_mean": float(conf.detach().mean().item()),
                "arousal_mean": float(arousal.detach().mean().item()),
                "conflict_mean": float(conflict.detach().mean().item()),
                "k": float(k),
                "want_replay": 1.0 if want_replay else 0.0,
                "cum_surprise": self.cum_surprise,
                "process_mode": 1.0 if process_mode == "system2" else 0.0,
                "deliberate": 1.0 if process_mode == "system2" else 0.0,
                "jspace_capacity": js_state.capacity_used,
                "jspace_n_active": float(js_state.n_active),
                "jspace_reportable": 1.0 if js_state.reportable else 0.0,
            }
        else:
            diag = {
                "k": float(k),
                "want_replay": 1.0 if want_replay else 0.0,
                "process_mode": 1.0 if process_mode == "system2" else 0.0,
                "jspace_capacity": js_state.capacity_used,
                "jspace_n_active": float(js_state.n_active),
            }

        if self.external_workspace and not use_global:
            # eval/serve: no running drive baseline between calls or users
            self.consciousness._bar_drive = None
        c_state = self.consciousness.step(
            jspace_capacity=js_state.capacity_used,
            jspace_n_active=js_state.n_active,
            jspace_reportable=js_state.reportable,
            broadcast=js_state.broadcast if hasattr(js_state, "broadcast") else None,
            arousal=arousal,
            conflict=conflict,
            confidence=conf,
            process_mode=process_mode,
            rel_S=rel_S,
            h=h,
        )
        if _full_diag:
            diag["consciousness_level"] = c_state.level
            diag["consciousness_ignition"] = 1.0 if c_state.ignition else 0.0
            diag["consciousness_phi"] = c_state.phi_proxy
            diag["consciousness_mode"] = {"unconscious": 0.0, "access": 0.5, "deliberate": 1.0}.get(c_state.mode, 0.0)

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
            process_mode=process_mode,
            deliberate=(process_mode == "system2"),
            jspace=js_state,
            jspace_broadcast=js_state.broadcast,
            consciousness=c_state,
        )

    # ------------------------------------------------------------------ J-space
    def _ensure_jspace(self, device: torch.device) -> None:
        if self._js_slots is None or self._js_slots.device != device:
            # random orthogonal-ish init for concept slots
            slots = torch.randn(self.jspace_slots, self.jspace_dim, device=device)
            slots = slots / (slots.norm(dim=-1, keepdim=True) + 1e-8)
            self._js_slots = slots
            self._js_occupancy = torch.zeros(self.jspace_slots, device=device)
            self._js_broadcast = torch.zeros(self.jspace_dim, device=device)

    def _update_jspace(
        self,
        rel_S: Tensor,
        conflict: Tensor,
        arousal: Tensor,
        conf: Tensor,
        process_mode: str,
        h: Optional[Tensor],
        device: torch.device,
        B: int,
    ) -> JSpaceState:
        """Update the capacity-limited global workspace and return its state."""
        self._ensure_jspace(device)

        # 1. Decay existing occupancy (forgetting / limited capacity pressure)
        self._js_occupancy = self._js_occupancy * self.jspace_decay

        # 2. Write strength driven by arousal + conflict (and stronger under System-2)
        write_strength = (0.6 * arousal.mean() + 0.4 * conflict.mean()).clamp(0.0, 3.0)
        if process_mode == "system2":
            write_strength = write_strength * 1.4

        # 3. Candidate content to write into workspace
        #    Prefer external field state h when available; otherwise synthesize from signals
        if h is not None:
            h_vec = h.detach()
            if h_vec.ndim == 2:
                h_vec = h_vec.mean(dim=0)          # (d_model,)
            # project / truncate to jspace_dim
            if h_vec.numel() >= self.jspace_dim:
                cand = h_vec[: self.jspace_dim]
            else:
                cand = torch.zeros(self.jspace_dim, device=device)
                cand[: h_vec.numel()] = h_vec
            cand = cand / (cand.norm() + 1e-8)
        else:
            # synthetic direction from current control signals
            seed = torch.tensor(
                [float(rel_S.mean()), float(conflict.mean()), float(arousal.mean()), float(conf.mean())],
                device=device,
            )
            cand = torch.zeros(self.jspace_dim, device=device)
            cand[: seed.numel()] = seed
            cand = cand / (cand.norm() + 1e-8)

        # 4. Soft write: find most similar slot or least occupied slot
        sims = torch.mv(self._js_slots, cand)                    # (n_slots,)
        best = int(sims.argmax().item())
        least = int(self._js_occupancy.argmin().item())

        if float(write_strength) >= self.jspace_write_threshold:
            # update best matching slot + boost occupancy
            alpha = min(0.6, 0.15 + 0.2 * float(write_strength))
            self._js_slots[best] = (1 - alpha) * self._js_slots[best] + alpha * cand
            self._js_slots[best] = self._js_slots[best] / (self._js_slots[best].norm() + 1e-8)
            self._js_occupancy[best] = (self._js_occupancy[best] + float(write_strength)).clamp(max=3.0)

            # if already crowded, also lightly touch the least-occupied slot
            if float(self._js_occupancy.mean()) > 0.8:
                self._js_occupancy[least] = (self._js_occupancy[least] + 0.3 * float(write_strength)).clamp(max=3.0)

        # 5. Enforce sparsity (Global Workspace has limited capacity)
        k_keep = max(1, int(self.jspace_slots * self.jspace_sparsity))
        topk_vals, topk_idx = torch.topk(self._js_occupancy, k=k_keep)
        mask = torch.zeros_like(self._js_occupancy)
        mask[topk_idx] = 1.0
        self._js_occupancy = self._js_occupancy * mask

        # 6. Broadcast = occupancy-weighted sum of active slots
        weights = self._js_occupancy / (self._js_occupancy.sum() + 1e-8)
        broadcast = torch.mv(self._js_slots.T, weights)          # (d,)
        self._js_broadcast = broadcast

        n_active = int((self._js_occupancy > 0.15).sum().item())
        capacity_used = float(self._js_occupancy.clamp(max=1.0).mean().item())
        reportable = n_active > 0 and capacity_used > 0.05

        # Expand to batch if needed for downstream convenience
        if B > 1:
            bcast_b = broadcast.unsqueeze(0).expand(B, -1)
            occ_b = self._js_occupancy.unsqueeze(0).expand(B, -1)
        else:
            bcast_b = broadcast
            occ_b = self._js_occupancy

        return JSpaceState(
            broadcast=bcast_b,
            occupancy=occ_b,
            capacity_used=capacity_used,
            n_active=n_active,
            reportable=reportable,
        )

    def _neutral_jspace(self, device: torch.device, B: int) -> JSpaceState:
        """Empty workspace state (learned workspace active): no state kept, no host syncs."""
        z = torch.zeros(self.jspace_dim, device=device)
        o = torch.zeros(self.jspace_slots, device=device)
        if B > 1:
            z = z.unsqueeze(0).expand(B, -1)
            o = o.unsqueeze(0).expand(B, -1)
        return JSpaceState(broadcast=z, occupancy=o, capacity_used=0.0, n_active=0, reportable=False)

    def read_jspace(self) -> Optional[JSpaceState]:
        """Return current workspace state (for logging / verbal report)."""
        if self._js_slots is None:
            return None
        weights = self._js_occupancy / (self._js_occupancy.sum() + 1e-8)
        broadcast = torch.mv(self._js_slots.T, weights)
        n_active = int((self._js_occupancy > 0.15).sum().item())
        return JSpaceState(
            broadcast=broadcast,
            occupancy=self._js_occupancy.clone(),
            capacity_used=float(self._js_occupancy.clamp(max=1.0).mean().item()),
            n_active=n_active,
            reportable=n_active > 0,
        )

    def inject_jspace(self, vector: Tensor, strength: float = 1.0) -> None:
        """
        Steer / inject a concept into the workspace (Anthropic-style intervention).
        vector: (d,) or (jspace_dim,) direction to add.
        """
        if self._js_slots is None:
            self._ensure_jspace(vector.device)
        v = vector.detach().flatten()
        if v.numel() < self.jspace_dim:
            pad = torch.zeros(self.jspace_dim, device=v.device)
            pad[: v.numel()] = v
            v = pad
        else:
            v = v[: self.jspace_dim]
        v = v / (v.norm() + 1e-8)

        # write into least occupied slot
        least = int(self._js_occupancy.argmin().item())
        alpha = min(0.8, 0.3 + 0.4 * strength)
        self._js_slots[least] = (1 - alpha) * self._js_slots[least] + alpha * v
        self._js_slots[least] = self._js_slots[least] / (self._js_slots[least].norm() + 1e-8)
        self._js_occupancy[least] = (self._js_occupancy[least] + strength).clamp(max=3.0)

    def clear_jspace(self) -> None:
        """Empty the workspace."""
        if self._js_occupancy is not None:
            self._js_occupancy.zero_()
        if self._js_broadcast is not None:
            self._js_broadcast.zero_()

    def export_jspace_state(self) -> Optional[dict]:
        """Snapshot rule-mode slot bank for session state (prevents cross-user leak)."""
        if self._js_slots is None:
            return None
        return {
            "js_slots": self._js_slots.detach().clone(),
            "js_occupancy": self._js_occupancy.detach().clone() if self._js_occupancy is not None else None,
            "js_broadcast": self._js_broadcast.detach().clone() if self._js_broadcast is not None else None,
            "last_process_mode": str(getattr(self, "_last_process_mode", "system1") or "system1"),
        }

    def import_jspace_state(self, pack: Optional[dict], device: Optional[torch.device] = None) -> None:
        """Restore slot bank from session state, or wipe for a new session."""
        if not pack:
            self._js_slots = None
            self._js_occupancy = None
            self._js_broadcast = None
            self._last_process_mode = "system1"
            return
        slots = pack.get("js_slots")
        if slots is None:
            self._js_slots = None
            self._js_occupancy = None
            self._js_broadcast = None
            self._last_process_mode = str(pack.get("last_process_mode") or "system1")
            return
        if not isinstance(slots, torch.Tensor):
            slots = torch.as_tensor(slots)
        if device is not None:
            slots = slots.to(device=device)
        self._js_slots = slots
        occ = pack.get("js_occupancy")
        if occ is None:
            self._js_occupancy = torch.zeros(self._js_slots.size(0), device=self._js_slots.device)
        else:
            if not isinstance(occ, torch.Tensor):
                occ = torch.as_tensor(occ, device=self._js_slots.device)
            self._js_occupancy = occ.to(device=self._js_slots.device)
        br = pack.get("js_broadcast")
        if br is None:
            self._js_broadcast = torch.zeros(self._js_slots.size(1), device=self._js_slots.device)
        else:
            if not isinstance(br, torch.Tensor):
                br = torch.as_tensor(br, device=self._js_slots.device)
            self._js_broadcast = br.to(device=self._js_slots.device)
        self._last_process_mode = str(pack.get("last_process_mode") or "system1")

    # ------------------------------------------------------------------ control
    @property
    def _use_global(self) -> bool:
        """Module-level running baselines are a TRAINING convenience only.

        In eval()/inference they are ignored for bar_S / replay counters.
        Rule-mode J-space slots still need session_jspace import/export (see model.forward)
        — module buffers alone would leak across users.
        """
        return self.track_global and getattr(self, "training", True)

    def dual_process(
        self,
        rel_S: Tensor,
        conflict: Optional[Tensor],
        confidence: Optional[Tensor] = None,
        *,
        force: Optional[str] = None,
    ) -> tuple[str, int]:
        """Dual-process control under metacognition (System-1 / System-2).

        Trade-off mitigation:
          - System-1 keeps k=1 (latency)
          - System-2 uses smooth k, not always k_max
          - light hysteresis avoids flicker S1↔S2 on noise
        """
        force_l = (force or "").strip().lower()
        if force_l in {"system1", "s1", "fast"}:
            self._last_process_mode = "system1"
            return "system1", 1
        if force_l in {"system2", "s2", "deliberate", "slow"}:
            self._last_process_mode = "system2"
            return "system2", max(1, int(self.k_max))

        s = float(rel_S.median().item()) if rel_S.numel() else 0.0
        c = 0.0 if conflict is None else float(conflict.median().item())
        conf = 1.0
        if confidence is not None and confidence.numel():
            conf = float(confidence.median().item())

        last = getattr(self, "_last_process_mode", "system1")
        # Accuracy bias: enter System-2 when surprise or conflict high, or confidence low
        enter_s2 = (s > 1.15) or (c > 1.1) or (conf < 0.42)
        # hysteresis: leave S2 only when signals calm down
        leave_s2 = (s < 0.75) and (c < 0.7) and (conf > 0.55)

        if last == "system2":
            if leave_s2:
                self._last_process_mode = "system1"
                return "system1", 1
            else:
                k = self._allocate_k(rel_S, conflict)
                return "system2", k
        else:
            if enter_s2:
                self._last_process_mode = "system2"
                k = self._allocate_k(rel_S, conflict)
                return "system2", k
            else:
                return "system1", 1

    def _allocate_k(self, rel_S: Tensor, conflict: Optional[Tensor]) -> int:
        """Map relative surprise (+ conflict) → integer field depth k ∈ [1, k_max]."""
        s = rel_S.median().item()
        c = 0.0 if conflict is None else float(conflict.median().item())
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
        delta = torch.zeros_like(rel_S)
        delta = torch.where(rel_S > 1.5, torch.full_like(delta, 0.5), delta)
        delta = torch.where(rel_S < 0.5, torch.full_like(delta, -0.5), delta)
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
        self._last_process_mode = "system1"
        self.bar_S = None
        self.bar_ent = None
        self.cum_surprise = 0.0
        self.tokens_seen = 0
        self.last_replay_at = 0
        self.clear_jspace()
        self.consciousness.reset()
        self._js_slots = None
        self._js_occupancy = None
        self._js_broadcast = None

    def state_dict(self) -> dict[str, Any]:
        d = {
            "bar_S": None if self.bar_S is None else float(self.bar_S.detach().cpu().item()),
            "bar_ent": None if self.bar_ent is None else float(self.bar_ent.detach().cpu().item()),
            "cum_surprise": self.cum_surprise,
            "tokens_seen": self.tokens_seen,
            "last_replay_at": self.last_replay_at,
        }
        if self._js_occupancy is not None:
            d["jspace_occupancy"] = self._js_occupancy.detach().cpu().tolist()
            d["jspace_n_active"] = int((self._js_occupancy > 0.15).sum().item())
        return d
