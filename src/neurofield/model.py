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
        # Token-parallel ops: embed/windows always; router+field+slow-read batched over T
        # when enabled. Delta-rule memory stays sequential. NEUROFIELD_PARALLEL_OPS=0 disables.
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
            "pending_pred": None,  # (legacy, unused: the prediction is recomputed from per-pass Hin)
            "k_depth": None,       # per-pass carry below is filled by forward()
            "Hin": None, "M_pass": None, "D_pass": None, "R_pass": None,
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
        collect_stats: bool = True,
    ) -> ModelOutput:
        """
        Parallel Form (main path): whole sequence + k depth passes.

        Sequential per-token time loop removed. Distributed (DDP) is unchanged —
        it only shards training across ranks; each rank still runs this forward.
        """
        B, T = input_ids.shape
        device = input_ids.device
        ab = self.ablation
        # per-position gate/memory norms cost O(T*k) tiny kernels; only when needed
        collect = bool(collect_stats or return_audit)
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

        # ---- per-depth-pass state carried ACROSS calls -------------------------------------------------
        # Training runs the whole sequence in one call; the server decodes ONE token per call with `state`.
        # For both to be the same function, every pass j must carry what position t-1 of that pass produced:
        #   Hin[j]  : the field state H that pass j READ at the last position  -> X_hat_{t-1} = f_p(Hin[j])
        #   M_pass  : the fast memory of pass j (each pass has its own M; it must keep accumulating)
        #   D_pass  : D of pass j at the last position  -> key of the next write
        #   R_pass  : read r of pass j at the last position -> field input at the next position
        # A state from another k_depth (or a fresh one) starts every pass from zero.
        carried = state.get("k_depth") == k_depth and state.get("M_pass") is not None
        Hin_prev = state["Hin"] if carried else [None] * k_depth
        M_prev = state["M_pass"] if carried else [None] * k_depth
        D_prev_pass = state["D_pass"] if carried else [None] * k_depth
        R_prev_pass = state["R_pass"] if carried else [None] * k_depth
        Hin_new: list[Tensor] = []
        M_new: list[Tensor] = []
        D_new: list[Tensor] = []
        R_new: list[Tensor] = []

        # slow-ring write cadence must match decoding: one snapshot per `segment_len` tokens (counted across calls)
        seg_cnt = int(state.get("seg_pos") or 0)
        snaps: list[tuple[Tensor, Tensor]] = []
        will_write_slow = bool(write_slow and ab.use_slow_memory and ab.use_fast_memory)

        for _j in range(k_depth):
            Hin_new.append(H[:, -1].detach())
            X_hat = self.predictor(H.reshape(B * T, -1)).view(B, T, -1)
            # prediction made at the PREVIOUS position: inside the call it is X_hat[:, t-1]; for the first
            # position it comes from the previous call's last position (zeros only at a sequence start).
            X_hat_first = (
                self.predictor(Hin_prev[_j]) if Hin_prev[_j] is not None else X_hat.new_zeros(B, X_hat.size(-1))
            )
            X_hat_shift = torch.cat([X_hat_first.unsqueeze(1), X_hat[:, :-1]], dim=1)
            E = X.detach() - X_hat_shift
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

            # Token-parallel router (positions independent given H)
            if self.parallel_ops:
                o_flat, pi_flat, _w = self.router(H.reshape(B * T, -1))
                O = o_flat.view(B, T, -1)
                pi = pi_flat.view(B, T, -1)
                ent_list = list((-(pi * pi.clamp_min(1e-12).log()).sum(-1).mean(0)).detach())
                bal_loss = bal_loss + (pi.mean(0) - 1.0 / max(1, self.cfg.n_skills)).pow(2).sum()  # mean over B → (T, n_skills); same as per-t loop
            else:
                o_list = []
                ent_list = []
                for t in range(T):
                    o_t, pi_t, _w = self.router(H[:, t])
                    o_list.append(o_t)
                    ent_list.append(-(pi_t * pi_t.clamp_min(1e-12).log()).sum(-1).mean().detach())
                    bal_loss = bal_loss + (
                        pi_t.mean(0) - 1.0 / max(1, self.cfg.n_skills)
                    ).pow(2).sum()
                O = torch.stack(o_list, dim=1)

            if ab.use_fast_memory:
                # Every depth pass owns its fast memory. Carry across decode calls.
                M = (
                    M_prev[_j]
                    if M_prev[_j] is not None
                    else torch.zeros(B, self.cfg.d_k, self.cfg.d_v, device=device, dtype=X.dtype)
                )
                d_prev0 = D_prev_pass[_j] if D_prev_pass[_j] is not None else None
                if self.parallel_ops:
                    # Parallel W_k/W_v/W_q; only rank-1 M update is sequential
                    M, R, more_snaps, seg_cnt = self.fast_mem.write_seq(
                        M, D, O, G,
                        d_prev0=d_prev0,
                        truncate_window=self._truncate_window,
                        max_memory_norm=float(self.safety.max_memory_norm or 0),
                        segment_len=self._segment_len if (will_write_slow and _j == k_depth - 1) else 0,
                        seg_cnt0=seg_cnt,
                        surprises=S if (will_write_slow and _j == k_depth - 1) else None,
                    )
                    snaps.extend(more_snaps)
                    if collect:
                        gates_audit.extend([G[:, t].detach().mean() for t in range(T)])
                        mem_norms.append(M.detach().norm())
                else:
                    reads = []
                    for t in range(T):
                        if t > 0:
                            d_prev = D[:, t - 1]
                        elif d_prev0 is not None:
                            d_prev = d_prev0
                        else:
                            d_prev = torch.zeros_like(D[:, 0])
                        M_before = M
                        if t > 0 and (t % max(1, self._truncate_window) == 0):
                            M = M.detach()
                        M = self.fast_mem.write(M, d_prev, O[:, t], G[:, t])
                        if self.safety.max_memory_norm > 0:
                            mn = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                            M = M * (self.safety.max_memory_norm / mn).clamp(max=1.0)
                        if collect:
                            write_norms.append((M - M_before).detach().norm())
                            mem_norms.append(M.detach().norm())
                            gates_audit.append(G[:, t].detach().mean())
                        reads.append(self.fast_mem.read(M, D[:, t]))
                        if will_write_slow and _j == k_depth - 1:
                            seg_cnt += 1
                            if seg_cnt >= self._segment_len:
                                snaps.append((M.detach(), S[:, t].detach()))
                                seg_cnt = 0
                    R = torch.stack(reads, dim=1)
            else:
                M = state.get("M")
                if M is None:
                    M = torch.zeros(B, self.cfg.d_k, self.cfg.d_v, device=device)
                R = torch.zeros(B, T, self.cfg.d_v, device=device, dtype=X.dtype)

            R_first = R_prev_pass[_j] if R_prev_pass[_j] is not None else R.new_zeros(B, R.size(-1))
            R_shift = torch.cat([R_first.unsqueeze(1), R[:, :-1]], dim=1)
            M_new.append(M.detach())
            D_new.append(D[:, -1].detach())
            R_new.append(R[:, -1].detach())
            if self.parallel_ops:
                H = self.field.single_step(
                    H.reshape(B * T, -1),
                    D.reshape(B * T, -1),
                    R_shift.reshape(B * T, -1),
                ).view(B, T, -1)
            else:
                H = torch.stack(
                    [
                        self.field.single_step(H[:, t], D[:, t], R_shift[:, t])
                        for t in range(T)
                    ],
                    dim=1,
                )

        if self.parallel_ops:
            o_flat, _pi, _w = self.router(H.reshape(B * T, -1))
            O = o_flat.view(B, T, -1)
        else:
            o_list = []
            for t in range(T):
                o_t, _pi, _w = self.router(H[:, t])
                o_list.append(o_t)
            O = torch.stack(o_list, dim=1)

        if ab.use_slow_memory:
            slow_slots = state.get("slow_slots")
            slow_filled = state.get("slow_filled")
            slow_ptr = int(state.get("slow_ptr") or 0)
            if self.parallel_ops:
                U = self.slow_mem.read_seq(D, slow_slots, slow_filled)
            else:
                U = torch.stack(
                    [
                        self.slow_mem.read(D[:, t], slots=slow_slots, filled=slow_filled)
                        for t in range(T)
                    ],
                    dim=1,
                )
            # Ring writes: a snapshot every `segment_len` tokens (same cadence whether the tokens arrive in one
            # training call or one per decode call) + one at the end of a multi-token call.
            if will_write_slow and T > 0:
                for M_snap, s_snap in snaps:
                    slow_slots, slow_ptr, slow_filled = self.slow_mem.write_slot(
                        slow_slots, slow_ptr, slow_filled, M_snap, weight=s_snap.clamp(0.5, 2.0)
                    )
                if T > 1 and seg_cnt > 0:
                    slow_slots, slow_ptr, slow_filled = self.slow_mem.write_slot(
                        slow_slots, slow_ptr, slow_filled, M, weight=S[:, -1].detach().clamp(0.5, 2.0)
                    )
                    seg_cnt = 0
                seg_pos = seg_cnt
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
        # Always report gate/mem so train logs are not stuck at gate=0.000
        if gates_audit:
            _gate_mean = float(
                torch.stack([g.detach().float().reshape(()) for g in gates_audit]).mean().item()
            )
        else:
            _gate_mean = float(G.detach().mean().item()) if T > 0 else 0.0
        if mem_norms:
            _mem_mean = float(
                torch.stack([x.detach().float().reshape(()) for x in mem_norms]).mean().item()
            )
        else:
            _mem_mean = float(M.detach().norm().item()) if isinstance(M, Tensor) else 0.0
        metrics: dict[str, float] = {
            "path": "parallel",
            "k_depth": float(k_depth),
            "gate_mean": _gate_mean,
            "mem_norm": _mem_mean,
            "mean_S": float(S.detach().mean().item()) if T > 0 else 0.0,
        }
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
                "ks": [float(k_depth)] * T,
                "router_entropy": [float(e) for e in ent_list],  # last pass, one per position
                "surprises": [float(s) for s in S.detach().mean(0)],
                "final_mem_norm": float(M.detach().norm()) if isinstance(M, Tensor) else 0.0,
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
            # per-pass carry (see above): makes one-call == chunked == token-by-token decode
            "k_depth": k_depth,
            "Hin": Hin_new,
            "M_pass": M_new,
            "D_pass": D_new,
            "R_pass": R_new,
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



