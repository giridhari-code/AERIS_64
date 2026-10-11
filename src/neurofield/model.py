"""NeuroField / AESC v2 core model — Parallel Form main path + §7 freeze."""

from __future__ import annotations

import os

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
        self.slow_mem = SlowMemory(
            cfg.d_model, cfg.d_k, cfg.d_v, n_slots=n_slots,
            compact_overflow=bool(getattr(cfg, "slow_compact", True)),
        )
        self._segment_len = segment
        # Token-parallel fast path (embedding, dendrite window, readout+logits batched over T).
        # Mathematically identical to the per-token path; NEUROFIELD_PARALLEL_OPS=0 disables it.
        self.parallel_ops = os.environ.get("NEUROFIELD_PARALLEL_OPS", "1") != "0"

        self.P_f = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        self.P_s = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        # Learnable scale so memory path can dominate when useful
        self.mem_scale = nn.Parameter(torch.tensor(1.5))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        # Fast path strong; slow context path also active (ring reads need P_s)
        nn.init.xavier_uniform_(self.P_f.weight, gain=2.0)
        nn.init.xavier_uniform_(self.P_s.weight, gain=1.0)
        with torch.no_grad():
            self.mem_scale.fill_(2.0)

        self.meta = Metacognition(k_max=cfg.k_max)
        self.drop = nn.Dropout(cfg.dropout)
        # Segment length = truncate window (fast M lifetime; ring write boundary)
        self._truncate_window = int(getattr(self.safety, "truncate_write_window", 32) or 32)
        self._segment_len = self._truncate_window
        # Slow query basis matches fast keys so context slots retrieve what was written
        self.slow_mem.tie_query_to(self.fast_mem.W_k)

    def set_ablation(self, ablation: AblationConfig) -> None:
        """Hot-swap ablation flags without rebuilding parameters (eval / contrib runs)."""
        self.ablation = ablation

    # ------------------------------------------------------------------
    # Per-user slow memory (the part of the state that may outlive a session)
    # ------------------------------------------------------------------
    def export_slow_state(self, state: Optional[dict]) -> Optional[dict[str, Tensor]]:
        """Slow ring of batch element 0 as plain CPU tensors (fast M / h / history are NOT included)."""
        if not state or state.get("slow_slots") is None:
            return None
        return {
            "slow_slots": state["slow_slots"][0].detach().float().cpu().contiguous(),
            "slow_filled": state["slow_filled"][0].detach().float().cpu().contiguous(),
            "slow_ptr": torch.tensor([int(state.get("slow_ptr") or 0)], dtype=torch.int64),
        }

    def new_state(self, batch: int, device: torch.device, slow: Optional[dict[str, Tensor]] = None) -> dict:
        """Fresh FAST state (M, h, dendrite history = zeros) optionally seeded with a saved slow ring.

        A saved ring whose shape does not match this model (different checkpoint) is ignored.
        """
        st = self._init_state(batch, device)
        if not slow:
            return st
        try:
            slots, filled = slow["slow_slots"], slow["slow_filled"]
            S, dk, dv = self.slow_mem.n_slots, self.cfg.d_k, self.cfg.d_v
            if tuple(slots.shape) != (S, dk, dv) or tuple(filled.shape) != (S,):
                return st
            st["slow_slots"] = slots.to(device).unsqueeze(0).repeat(batch, 1, 1, 1)
            st["slow_filled"] = filled.to(device).unsqueeze(0).repeat(batch, 1)
            st["slow_ptr"] = int(slow["slow_ptr"].reshape(-1)[0].item()) % S
        except (KeyError, RuntimeError, ValueError):
            return self._init_state(batch, device)
        return st

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
            "seg_pos": 0,  # tokens since the last slow-ring write (global, survives 1-token calls)
        }


    def forward(
        self,
        input_ids: Tensor,               # (B, T)
        targets: Optional[Tensor] = None,
        state: Optional[dict] = None,
        return_audit: bool = False,
        lambda_err: float = 0.1,
        lambda_bal: float = 0.01,
        write_slow: bool = True,
        k_depth: Optional[int] = None,
    ) -> ModelOutput:
        """
        Parallel Form (main path): whole sequence + k depth passes.

        Sequential per-token time loop removed. Distributed (DDP) is unchanged —
        it only shards training across ranks; each rank still runs this forward.
        """
        B, T = input_ids.shape
        device = input_ids.device
        ab = self.ablation
        if state is None:
            state = self._init_state(B, device)

        k_depth = int(k_depth or max(1, int(self.cfg.k_max)))
        if not ab.use_metacognition:
            k_depth = max(1, int(ab.fixed_k))

        X = self.drop(self.embed(input_ids))  # (B, T, D)
        W = self.dendrite.window
        x_hist = state.get("x_hist")
        if x_hist is None or x_hist.shape != (B, W, X.size(-1)):
            x_hist = X.new_zeros(B, W, X.size(-1))
        Xp = torch.cat([x_hist, X], dim=1)
        D_win = self.dendrite.window_features(Xp, T)  # (B, T, D)

        H = D_win.clone()
        R = torch.zeros(B, T, self.cfg.d_v, device=device, dtype=X.dtype)
        err_energy = X.new_zeros(())
        bal_loss = X.new_zeros(())
        gates_audit: list[Tensor] = []
        write_norms: list[Tensor] = []
        mem_norms: list[Tensor] = []

        def _shift(x: Tensor) -> Tensor:
            return torch.cat([x.new_zeros(x[:, :1].shape), x[:, :-1]], dim=1)

        frozen = bool(state.get("memory_frozen"))

        for _j in range(k_depth):
            X_hat = self.predictor(H.reshape(B * T, -1)).view(B, T, -1)
            E = X.detach() - _shift(X_hat)
            err_energy = err_energy + E.pow(2).mean()
            S = E.abs().mean(dim=-1)  # (B, T)

            if ab.use_neuromodulator:
                activity = H.abs().mean(dim=-1)
                logit = (
                    self.neuromod.a_s * (S - 0.5)
                    + self.neuromod.a_c * activity
                    + self.neuromod.bias
                )
                G = torch.sigmoid(logit).unsqueeze(-1).clamp(max=self.safety.gate_cap)
            else:
                G = torch.full(
                    (B, T, 1), float(ab.fixed_gate), device=device, dtype=X.dtype
                ).clamp(max=self.safety.gate_cap)
            if frozen:
                G = torch.zeros_like(G)

            D = D_win + self.dendrite.gamma * E

            o_list = []
            for t in range(T):
                o_t, pi_t, _w = self.router(H[:, t])
                o_list.append(o_t)
                bal_loss = bal_loss + (
                    pi_t.mean(0) - 1.0 / max(1, self.cfg.n_skills)
                ).pow(2).sum()
            O = torch.stack(o_list, dim=1)

            if ab.use_fast_memory:
                # Fresh M each depth (leakage-safe across positions)
                M = torch.zeros(
                    B, self.cfg.d_k, self.cfg.d_v, device=device, dtype=X.dtype
                )
                # Optional: warm-start depth0 from session state for T=1 serve continuity
                if _j == 0 and state.get("M") is not None and T == 1:
                    M = state["M"]
                reads = []
                for t in range(T):
                    d_prev = D[:, t - 1] if t > 0 else (
                        state.get("d_prev")
                        if state.get("d_prev") is not None
                        else torch.zeros_like(D[:, 0])
                    )
                    if t == 0 and state.get("d_prev") is not None and d_prev.shape == D[:, 0].shape:
                        d_prev = state["d_prev"]
                    elif t == 0:
                        d_prev = torch.zeros_like(D[:, 0])
                    M_before = M
                    M = self.fast_mem.write(M, d_prev, O[:, t], G[:, t])
                    if self.safety.max_memory_norm > 0:
                        mn = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                        M = M * (self.safety.max_memory_norm / mn).clamp(max=1.0)
                    write_norms.append((M - M_before).detach().norm())
                    mem_norms.append(M.detach().norm())
                    reads.append(self.fast_mem.read(M, D[:, t]))
                    gates_audit.append(G[:, t].detach().mean())
                R = torch.stack(reads, dim=1)
            else:
                M = state.get("M")
                if M is None:
                    M = torch.zeros(B, self.cfg.d_k, self.cfg.d_v, device=device)
                R = torch.zeros(B, T, self.cfg.d_v, device=device, dtype=X.dtype)

            R_shift = _shift(R)
            H = torch.stack(
                [
                    self.field.single_step(H[:, t], D[:, t], R_shift[:, t])
                    for t in range(T)
                ],
                dim=1,
            )

        o_list = []
        for t in range(T):
            o_t, _pi, _w = self.router(H[:, t])
            o_list.append(o_t)
        O = torch.stack(o_list, dim=1)

        if ab.use_slow_memory:
            slow_slots = state.get("slow_slots")
            slow_filled = state.get("slow_filled")
            slow_ptr = int(state.get("slow_ptr") or 0)
            U = torch.stack(
                [
                    self.slow_mem.read(D[:, t], slots=slow_slots, filled=slow_filled)
                    for t in range(T)
                ],
                dim=1,
            )
            # segment write at end of chunk
            if write_slow and T > 0 and ab.use_fast_memory:
                seg_pos = int(state.get("seg_pos") or 0) + T
                if seg_pos >= self._segment_len or T > 1:
                    w = S[:, -1].detach().clamp(0.5, 2.0)
                    slow_slots, slow_ptr, slow_filled = self.slow_mem.write_slot(
                        slow_slots, slow_ptr, slow_filled, M, weight=w
                    )
                    seg_pos = 0
            else:
                seg_pos = int(state.get("seg_pos") or 0)
        else:
            U = torch.zeros(B, T, self.cfg.d_v, device=device, dtype=X.dtype)
            slow_slots = state.get("slow_slots")
            slow_filled = state.get("slow_filled")
            slow_ptr = int(state.get("slow_ptr") or 0)
            seg_pos = int(state.get("seg_pos") or 0)

        logits = self.head(
            self.out_norm(O + self.mem_scale * self.P_f(R) + self.P_s(U))
        )

        loss = None
        metrics: dict[str, float] = {"path": "parallel", "k_depth": float(k_depth)}
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss = (
                ce
                + lambda_err * (err_energy / max(1, k_depth))
                + lambda_bal * (bal_loss / max(1, T * k_depth))
            )
            metrics.update(
                {
                    "ce": float(ce.detach().item()),
                    "err_energy": float((err_energy / max(1, k_depth)).detach().item()),
                }
            )

        audit = None
        if return_audit:
            audit = {
                "path": "parallel",
                "gates": [float(g) for g in gates_audit[-T:]] if gates_audit else [],
                "write_norms": [float(x) for x in write_norms[-T:]] if write_norms else [],
                "mem_norms": [float(x) for x in mem_norms[-T:]] if mem_norms else [],
            }

        new_state = {
            **state,
            "h": H[:, -1].detach() if T else state.get("h"),
            "M": M.detach() if isinstance(M, Tensor) else state.get("M"),
            "r": R[:, -1].detach() if T else state.get("r"),
            "error": E[:, -1].detach() if T else state.get("error"),
            "x_hist": Xp[:, T:].detach(),
            "d_prev": D[:, -1].detach() if T else state.get("d_prev"),
            "pending_pred": None,
            "slow_slots": slow_slots,
            "slow_filled": slow_filled,
            "slow_ptr": slow_ptr,
            "seg_pos": seg_pos,
            "memory_frozen": frozen,
        }
        return ModelOutput(
            logits=logits, loss=loss, metrics=metrics, audit=audit, state=new_state
        )

    def replay_slow(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        self.slow_mem.replay(M, surprises, eta)

    def reset_meta(self) -> None:
        self.meta.reset()

    def freeze_writes(self) -> None:
        """Legacy hard freeze of neuromodulator (prefer state memory_frozen → g=0)."""
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



