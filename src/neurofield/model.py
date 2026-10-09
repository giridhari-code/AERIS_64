"""NeuroField v2 core model – production implementation matching Appendix A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.modules import (
    Dendrite,
    NeuralField,
    Neuromodulator,
    SkillRouter,
    FastMemory,
    SlowMemory,
    Predictor,
    Metacognition,
)


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Optional[Tensor] = None
    metrics: dict[str, float] = field(default_factory=dict)
    audit: Optional[dict[str, Any]] = None
    state: Optional[dict[str, Tensor]] = None


def _tolist_t(xs: list[Tensor]) -> list[float]:
    """One device->host transfer for a list of 0-dim tensors."""
    if not xs:
        return []
    return torch.stack([x.detach().float().reshape(()) for x in xs]).tolist()


def _mean_t(xs: list[Tensor]) -> float:
    if not xs:
        return 0.0
    return float(torch.stack([x.detach().float().reshape(()) for x in xs]).mean().item())


class NeuroField(nn.Module):
    """
    NeuroField / AESC — two-speed memory language model.

    =====================================================================
    PER-TOKEN LOOP (every word / token t = 0 .. T-1)
    =====================================================================

        x_t  →  Embedding
             →  Dendrite(x_hist, error)     →  d_t   (changes every token)
             →  Metacognition               →  k, gate_bias
             →  Neural Field(h, d_t, r, k)  →  h     (k recurrent steps)
             →  Predictor(h)                →  pending pred for x_{t+1}
             →  Router / Skills(h)          →  o_t
             →  Neuromodulator              →  g
             →  FastMemory write/read       →  M, r
             →  SlowMemory read (ring)      →  u
             →  logits_t

        Next token: same loop again — d_t is recomputed each step
        (dendrite window + delayed error), so the field always sees
        a fresh local feature, not a static context vector.

    =====================================================================
    TWO-SPEED MEMORY
    =====================================================================

    Fast memory M:
        Short segment. Delta-rule write every token (key=d_{t-1}, value=o_t).
        Survives ~segment_len tokens (truncate window).

    Slow memory = context-window ring of slots:
        n_slots ≈ context_window / segment_len
        At each segment boundary: snapshot fast M → next ring slot.
        Read: content-address over all filled slots (long context lives here).

    Timing (leakage-safe):
      - Predictor forecasts x(t+1) from state after token t.
      - Error is used only from step t+1 onward.
      - Nothing from x(t+1) reaches the logits that predict x(t+1).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        safety: SafetyConfig | None = None,
        ablation: AblationConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.safety = safety or SafetyConfig()
        self.ablation = ablation or AblationConfig()

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dendrite = Dendrite(cfg.d_model, window=cfg.dendrite_window)
        self.field = NeuralField(cfg.d_model, d_mem=cfg.d_v)
        self.predictor = Predictor(cfg.d_model)
        self.neuromod = Neuromodulator(cfg.d_model)
        self.router = SkillRouter(cfg.d_model, cfg.n_skills, cfg.top_k)
        self.fast_mem = FastMemory(cfg.d_model, cfg.d_k, cfg.d_v)
        # Context window ring: n_slots ≈ max_seq_len / segment_len
        segment = max(1, int(self.safety.truncate_write_window))
        n_slots = int(getattr(cfg, "context_slots", 0) or 0)
        if n_slots <= 0:
            n_slots = max(4, (cfg.max_seq_len + segment - 1) // segment)
        self.slow_mem = SlowMemory(cfg.d_model, cfg.d_k, cfg.d_v, n_slots=n_slots)
        self._segment_len = segment

        self.P_f = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        self.P_s = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        # Learnable scale so memory path can dominate when useful
        self.mem_scale = nn.Parameter(torch.tensor(1.5))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        # Fast-path strong; slow-path quiet until replay fills M_s (was xavier noise).
        nn.init.xavier_uniform_(self.P_f.weight, gain=2.0)
        # Slow context path must be able to contribute (was zeros → dead ring)
        nn.init.xavier_uniform_(self.P_s.weight, gain=1.0)
        # mem_scale starts able to dominate skills when recall needs it
        with torch.no_grad():
            self.mem_scale.fill_(2.0)

        self.meta = Metacognition(k_max=cfg.k_max)
        self.drop = nn.Dropout(cfg.dropout)
        # Truncated BPTT window (overridable via safety or default 32 per paper config)
        self._truncate_window = int(getattr(self.safety, "truncate_write_window", 32) or 32)
        if not getattr(cfg, "context_slots", None):
            self.slow_mem.n_slots = max(4, cfg.max_seq_len // max(1, self._truncate_window))
        # Slow query basis matches fast keys so context slots retrieve what was written
        self.slow_mem.tie_query_to(self.fast_mem.W_k)

    def set_ablation(self, ablation: AblationConfig) -> None:
        """Hot-swap ablation flags without rebuilding parameters (eval / contrib runs)."""
        self.ablation = ablation

    def _init_state(self, batch: int, device: torch.device) -> dict[str, Tensor]:
        return {
            "h": torch.zeros(batch, self.cfg.d_model, device=device),
            "M": torch.zeros(batch, self.cfg.d_k, self.cfg.d_v, device=device),
            "error": torch.zeros(batch, self.cfg.d_model, device=device),
            "r": torch.zeros(batch, self.cfg.d_v, device=device),
            "d_prev": torch.zeros(batch, self.cfg.d_model, device=device),
            "x_hist": torch.zeros(
                batch, self.cfg.dendrite_window, self.cfg.d_model, device=device
            ),
            "pending_pred": None,  # prediction made at t for x_{t+1}
            "bar_S": None,  # per-session running surprise (isolation)
            "last_router_ent": None,  # prior-step conflict for metacognition
            # Context window ring in slow memory
            "slow_slots": torch.zeros(
                batch, self.slow_mem.n_slots, self.cfg.d_k, self.cfg.d_v, device=device
            ),
            "slow_ptr": 0,
            "slow_filled": torch.zeros(batch, self.slow_mem.n_slots, device=device),
        }

    def forward(
        self,
        input_ids: Tensor,               # (B, T)
        targets: Optional[Tensor] = None,
        state: Optional[dict] = None,
        return_audit: bool = False,
        lambda_err: float = 0.1,
        lambda_bal: float = 0.01,
    ) -> ModelOutput:
        B, T = input_ids.shape
        device = input_ids.device

        if state is None:
            state = self._init_state(B, device)

        h = state["h"]
        M = state["M"]
        error = state["error"]
        r = state["r"]
        d_prev = state["d_prev"]
        x_hist = state["x_hist"]
        pending_pred = state.get("pending_pred")
        bar_S = state.get("bar_S")  # per-session surprise baseline
        last_router_ent = state.get("last_router_ent")
        slow_slots = state.get("slow_slots")
        slow_ptr = int(state.get("slow_ptr") or 0)
        slow_filled = state.get("slow_filled")
        if slow_slots is None:
            slow_slots, slow_ptr, slow_filled = self.slow_mem.init_slots(B, device)

        logits_list: list[Tensor] = []
        err_energy = torch.tensor(0.0, device=device)
        bal_loss = torch.tensor(0.0, device=device)
        all_S: list[Tensor] = []
        # Per-step diagnostics are kept as 0-dim tensors and converted once at the
        # end of forward (no per-token .item() => no per-token GPU sync).
        gates: list[Tensor] = []
        write_norms: list[Tensor] = []
        mem_norms: list[Tensor] = []
        ks: list[int] = []
        router_ents: list[Tensor] = []

        for t in range(T):
            x_t = self.drop(self.embed(input_ids[:, t]))

            # Shift causal history
            x_hist = torch.cat([x_hist[:, 1:], x_t.unsqueeze(1)], dim=1)

            # Delayed error arrives now (from previous prediction)
            if pending_pred is not None:
                error, S = self.predictor.compute_error_and_surprise(pending_pred, x_t)
                all_S.append(S)
                err_energy = err_energy + error.pow(2).mean()
            else:
                S = torch.ones(B, device=device)

            # Dendrite: local window + delayed error → d_t (fresh every token)
            d_t = self.dendrite(x_hist, error)

            # Metacognition: monitor (surprise, conflict) → control (k, gate_bias)
            ab = self.ablation
            gate_bias = None
            if ab.use_metacognition:
                ctrl = self.meta.step(
                    S=S,
                    bar_S=bar_S,
                    router_entropy=last_router_ent,
                    mem_norm=M.detach().norm(dim=(-2, -1)) if M is not None else None,
                )
                rel_S, bar_S, k = ctrl.rel_S, ctrl.bar_S, ctrl.k
                gate_bias = ctrl.gate_bias
            else:
                rel_S = torch.ones(B, device=device)
                k = max(1, int(ab.fixed_k))
            ks.append(k)

            # Neural field: k recurrent steps on this token's d_t (d_t is new every t)
            h = self.field(h, d_t, r, k)

            # Predict next embedding (compared when x_{t+1} arrives — delayed error)
            pending_pred = self.predictor(h)

            # Skills + router
            o_t, pi, _ = self.router(h)
            bal_loss = bal_loss + (pi.mean(0) - 1.0 / self.cfg.n_skills).pow(2).sum()
            step_ent = (-(pi * (pi + 1e-8).log()).sum(-1)).detach()  # (B,)
            router_ents.append(step_ent.mean())
            last_router_ent = step_ent  # feed conflict to metacognition next step

            # Gate (neuromod + metacognitive encoding bias)
            if ab.use_neuromodulator:
                g = self.neuromod(
                    rel_S, h, gate_cap=self.safety.gate_cap, gate_bias=gate_bias
                )
            else:
                g = torch.full((B, 1), float(ab.fixed_gate), device=device)
                g = g.clamp(max=self.safety.gate_cap)
            gates.append(g.detach().mean())

            # Fast write (truncated BPTT) — paper: key=d_{t-1}, value=W_v o_t
            M_before = M
            if ab.use_fast_memory:
                if t > 0 and (t % self._truncate_window != 0):
                    M = self.fast_mem.write(M, d_prev, o_t, g)
                else:
                    M = self.fast_mem.write(M.detach(), d_prev, o_t, g)

                # Write-norm audit + hard cap (paper Section 6)
                delta_M = (M - M_before).detach()
                # Per-sequence write norm (B,) — not batch-global scalar.
                # Avoids one sequence's large write throttling the whole batch
                # and keeps training (B>1) consistent with serving (B=1).
                wn_b = delta_M.norm(dim=(-2, -1))  # (B,)
                write_norms.append(wn_b.mean())
                if self.safety.max_write_norm > 0:
                    scale_w = (self.safety.max_write_norm / (wn_b + 1e-8)).clamp(max=1.0)
                    scale_w = scale_w.view(-1, 1, 1)
                    M = M_before + scale_w * (M - M_before)

                mem_norms.append(M.detach().norm())

                # Safety memory-norm cap
                if self.safety.max_memory_norm > 0:
                    mn = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                    scale = (self.safety.max_memory_norm / mn).clamp(max=1.0)
                    M = M * scale

                r = self.fast_mem.read(M, d_t)
                # Segment end (or last token) → fast M snapshot into slow context ring
                at_segment = (t > 0 and (t % self._truncate_window == 0))
                at_eos = (t == T - 1)
                if at_segment or at_eos:
                    w = all_S[-1] if all_S else torch.ones(B, device=device)
                    slow_slots, slow_ptr, slow_filled = self.slow_mem.write_slot(
                        slow_slots, slow_ptr, slow_filled, M, weight=w
                    )
            else:
                write_norms.append(torch.tensor(0.0, device=device))
                mem_norms.append(M.detach().norm())
                r = torch.zeros(B, self.cfg.d_v, device=device)

            if ab.use_slow_memory:
                u = self.slow_mem.read(d_t, slots=slow_slots, filled=slow_filled)
            else:
                u = torch.zeros(B, self.cfg.d_v, device=device)

            # Output — memory path scaled so it can dominate
            combined = self.out_norm(
                o_t + self.mem_scale * self.P_f(r) + self.P_s(u)
            )
            logits_t = self.head(combined)
            logits_list.append(logits_t)

            d_prev = d_t

        logits = torch.stack(logits_list, dim=1)

        loss = None
        metrics: dict[str, float] = {}
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            total = (
                ce
                + lambda_err * (err_energy / max(1, T - 1))
                + lambda_bal * (bal_loss / T)
            )
            loss = total
            # Per-sequence mean surprise for paper-faithful slow replay
            if all_S:
                stacked = torch.stack(all_S, dim=0)  # (T', B)
                seq_surprise = stacked.mean(0)  # (B,)
            else:
                seq_surprise = torch.ones(B, device=device)
            metrics = {
                "ce": ce.item(),
                "err_energy": (err_energy / max(1, T - 1)).item(),
                "bal": (bal_loss / T).item(),
                "gate_mean": _mean_t(gates),
                "mem_norm": _mean_t(mem_norms),
                "k_mean": sum(ks) / max(1, len(ks)),
                "router_ent": _mean_t(router_ents),
                "mean_S": float(seq_surprise.mean().item()),
            }
            # Stash for trainer replay (not serialized as metric float only)
            metrics["_seq_surprise"] = seq_surprise.detach()

        new_state = {
            "h": h.detach(),
            "M": M.detach(),
            "error": error.detach(),
            "r": r.detach(),
            "d_prev": d_prev.detach(),
            "x_hist": x_hist.detach(),
            "pending_pred": pending_pred.detach() if pending_pred is not None else None,
            "bar_S": bar_S.detach() if isinstance(bar_S, Tensor) else bar_S,
            "last_router_ent": last_router_ent.detach() if isinstance(last_router_ent, Tensor) else last_router_ent,
            "slow_slots": slow_slots.detach() if isinstance(slow_slots, Tensor) else slow_slots,
            "slow_ptr": slow_ptr,
            "slow_filled": slow_filled.detach() if isinstance(slow_filled, Tensor) else slow_filled,
        }

        audit = None
        if return_audit or self.safety.enable_audit:
            audit = {
                "gates": _tolist_t(gates),
                "write_norms": _tolist_t(write_norms),
                "mem_norms": _tolist_t(mem_norms),
                "ks": ks,
                "router_entropy": _tolist_t(router_ents),
                "surprises": _tolist_t([s.mean() for s in all_S]),
                "final_mem_norm": float(M.detach().norm()),
            }

        return ModelOutput(
            logits=logits,
            loss=loss,
            metrics=metrics,
            audit=audit,
            state=new_state,
        )

    def replay_slow(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        self.slow_mem.replay(M, surprises, eta)

    def reset_meta(self) -> None:
        self.meta.reset()

    def freeze_writes(self) -> None:
        for p in self.neuromod.parameters():
            p.requires_grad_(False)
        self.neuromod.a_s.data.zero_()
        self.neuromod.a_c.data.zero_()
        self.neuromod.bias.data.fill_(-20.0)

    def param_report(self) -> dict[str, int]:
        """Count unique parameters (tied embed/head counted once)."""
        report: dict[str, int] = {}
        seen: set[int] = set()
        unique_total = 0
        for name, module in self.named_children():
            n = 0
            for p in module.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    n += p.numel()
                    unique_total += p.numel()
            report[name] = n
        report["total"] = unique_total
        report["total_with_tied_doublecount"] = sum(
            sum(p.numel() for p in m.parameters()) for _, m in self.named_children()
        )
        return report
