"""SENTINEL core model (two-speed memory architecture) — Parallel Form main path + §7 freeze."""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Inference-friendly defaults (safe; does not change numerics of the architecture)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sentinel.config import AblationConfig, ModelConfig, SafetyConfig
from sentinel.modules import (
    Dendrite,
    NeuralField,
    Neuromodulator,
    SkillRouter,
    FastMemory,
    SlowMemory,
    HierarchicalPredictor,
    Predictor,
    Metacognition,
    GlobalWorkspace,
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


class SENTINEL(nn.Module):
    """
    SENTINEL — two-speed memory + Astra-style looped recurrent depth language model.

    =====================================================================
    PER-TOKEN LOOP (every word / token t = 0 .. T-1)
    =====================================================================

        x_t  →  Embedding
             →  Dendrite(x_hist, error)     →  d_t   (past-W window only; no x_t)
             →  Metacognition (+ J-space)   →  k, gate_bias, workspace
             →  Neural Field(h, d_t, r, k)  →  h     (k recurrent steps)
             →  Predictor(h)                →  pending pred for x_{t+1}
             →  Router / Skills(h)          →  o_t
             →  Neuromodulator              →  g
             →  FastMemory write/read       →  M, r
             →  SlowMemory read (ring)      →  u
             →  logits_t

        Next token: same loop again — d_t is recomputed each step from the
        causal past window + delayed error (current x_t is NOT in the window).
        Parallel (T>1) and decode (T=1) paths are identical.

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
        self.field = NeuralField(
            cfg.d_model,
            d_mem=cfg.d_v,
            reinject_input=bool(cfg.field_reinject_input),
            loop_residual_scale=bool(cfg.field_loop_residual_scale),
            jspace_dim=int(cfg.jspace_dim),
        )
        if getattr(cfg, "hierarchical_predict", True):
            self.predictor = HierarchicalPredictor(
                cfg.d_model,
                mid_horizon=int(getattr(cfg, "pred_mid_horizon", 4) or 4),
                coarse_horizon=int(getattr(cfg, "pred_coarse_horizon", 16) or 16),
            )
        else:
            self.predictor = Predictor(cfg.d_model)
        self.neuromod = Neuromodulator(cfg.d_model)
        self.router = SkillRouter(
            cfg.d_model,
            cfg.n_skills,
            cfg.top_k,
            jspace_dim=int(cfg.jspace_dim),
            sparse_skill_exec=bool(getattr(cfg, "sparse_skill_exec", True)),
        )
        self.fast_mem = FastMemory(cfg.d_model, cfg.d_k, cfg.d_v)
        # Context window ring: n_slots ≈ max_seq_len / segment_len
        segment = max(1, int(self.safety.truncate_write_window))
        n_slots = int(cfg.context_slots or 0)
        if n_slots <= 0:
            n_slots = max(4, (cfg.max_seq_len + segment - 1) // segment)
        self.slow_mem = SlowMemory(
            cfg.d_model, cfg.d_k, cfg.d_v, n_slots=n_slots,
            compact_overflow=bool(cfg.slow_compact),
        )
        self._segment_len = segment
        # Token-parallel ops: embed/windows always; router+field+slow-read batched over T
        # when enabled. Delta-rule memory stays sequential. NEUROFIELD_PARALLEL_OPS=0 disables.
        self.parallel_ops = True  # looped form only; token-serial outer path removed

        self.P_f = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        self.P_s = nn.Linear(cfg.d_v, cfg.d_model, bias=False)
        # Learnable scales so BOTH memory paths can contribute (slow was under-scaled)
        self.mem_scale = nn.Parameter(torch.tensor(2.0))
        self.slow_scale = nn.Parameter(torch.tensor(1.5))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        # Balanced init: slow ring must not be a dead path under short-seq training
        nn.init.xavier_uniform_(self.P_f.weight, gain=2.0)
        nn.init.xavier_uniform_(self.P_s.weight, gain=1.5)
        with torch.no_grad():
            self.mem_scale.fill_(2.0)
            self.slow_scale.fill_(1.5)

        # Learned global workspace (opt-in). "rule" = legacy slot bank, old state_dict keys unchanged.
        _ws_learned_cfg = (
            str(getattr(cfg, "workspace_mode", "rule") or "rule").lower() == "learned"
            and int(cfg.jspace_dim) > 0
        )
        self.meta = Metacognition(
            external_workspace=_ws_learned_cfg,
            k_max=cfg.k_max,
            jspace_slots=int(cfg.jspace_slots),
            jspace_dim=int(cfg.jspace_dim),
            jspace_decay=float(cfg.jspace_decay),
            jspace_write_threshold=float(cfg.jspace_write_threshold),
            jspace_sparsity=float(cfg.jspace_sparsity),
        )
        # Consciousness thresholds from config (if simulator present)
        if hasattr(self.meta, "consciousness") and self.meta.consciousness is not None:
            c = self.meta.consciousness
            c.ignition_threshold = float(cfg.consciousness_ignition_threshold)
            c.access_threshold = float(cfg.consciousness_access_threshold)
            c.deliberate_threshold = float(cfg.consciousness_deliberate_threshold)
        self._last_jspace_broadcast = None
        self._last_consciousness = None
        if _ws_learned_cfg:
            self.workspace = GlobalWorkspace(
                cfg.d_model,
                int(cfg.jspace_dim),
                n_atoms=int(cfg.ws_atoms),
                k_active=int(cfg.ws_k_active),
                rank=int(cfg.ws_rank),
            )
        else:
            self.workspace = None
        self._ws_steer: Optional[Tensor] = None   # optional (n_atoms,) steering vector (experiments)
        self._last_ws_code: Optional[Tensor] = None  # detached code of the last read (interpretability)
        self.drop = nn.Dropout(cfg.dropout)
        # BPTT truncate for fast-M scan (can be long). Slow ring writes MORE often
        # so the context path actually fills during typical train seq lengths.
        self._truncate_window = int(getattr(self.safety, "truncate_write_window", 32) or 32)
        slow_seg = int(getattr(self.safety, "slow_segment_len", 0) or 0)
        if slow_seg <= 0:
            # default: 1/4 of truncate, at least 4, at most 16 — more ring snaps
            slow_seg = max(4, min(16, self._truncate_window // 4 or 4))
        self._segment_len = max(1, slow_seg)
        # Slow query basis matches fast keys so context slots retrieve what was written
        self.slow_mem.tie_query_to(self.fast_mem.W_k)

    def set_ablation(self, ablation: AblationConfig) -> None:
        """Hot-swap ablation flags without rebuilding parameters (eval / contrib runs)."""
        self.ablation = ablation

    # ------------------------------------------------------------------
    # Per-user slow memory (the part of the state that may outlive a session)
    # ------------------------------------------------------------------
    def export_slow_state(self, state: Optional[dict]) -> Optional[dict[str, Tensor]]:
        """Slow ring of batch element 0 as plain CPU tensors (fast M / h / history are NOT included).

        Self-heals before export so corrupt long-session state is not persisted.
        """
        if not state or state.get("slow_slots") is None:
            return None
        state = self.heal_session_state(state) or state
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
            slots_b = slots.to(device).unsqueeze(0).repeat(batch, 1, 1, 1)
            filled_b = filled.to(device).unsqueeze(0).repeat(batch, 1)
            # Self-heal on load (corrupt checkpoints / long sessions)
            slots_b, filled_b, _ = self.slow_mem.heal_state(slots_b, filled_b, None)
            st["slow_slots"] = slots_b
            st["slow_filled"] = filled_b
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
            "last_S": None,  # (B,) last mean surprise — metacognition input
            "last_router_ent": None,  # prior-step conflict for metacognition
            "want_replay": False,
            # Context window ring in slow memory
            "slow_slots": torch.zeros(
                batch, self.slow_mem.n_slots, self.cfg.d_k, self.cfg.d_v, device=device
            ),
            "slow_ptr": 0,
            "slow_filled": torch.zeros(batch, self.slow_mem.n_slots, device=device),
            "seg_pos": 0,  # tokens since the last slow-ring write (global, survives 1-token calls)
        }


    # J-lens (verbalizability) loss: the workspace read at mid-depth must be able to "say" the
    # next token and tokens a few steps ahead (paper: patterns that make a word more likely
    # to be said at some point in the future). Subsampled positions keep the head cost small.
    _LENS_OFFSETS = (0, 1, 3)
    _LENS_STRIDE = 4

    def _lens_loss(self, H: Tensor, targets: Tensor) -> Tensor:
        B, T, _ = H.shape
        z = self.workspace.lens_state(H)  # norm(J h)
        off0 = int(torch.randint(0, self._LENS_STRIDE, (1,)).item())  # CPU generator, no GPU sync
        total = z.new_zeros(())
        used = 0
        for o in self._LENS_OFFSETS:
            if T - o <= off0:
                continue
            zi = z[:, off0 : T - o : self._LENS_STRIDE]
            ti = targets[:, off0 + o : T : self._LENS_STRIDE]
            n = min(zi.size(1), ti.size(1))
            if n <= 0:
                continue
            zi, ti = zi[:, :n], ti[:, :n]
            valid = (ti != -100).sum().clamp(min=1)
            ce = F.cross_entropy(
                self.head(zi).reshape(-1, self.cfg.vocab_size).float(),
                ti.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total = total + ce / valid
            used += 1
        return total / max(1, used)

    def forward(
        self,
        input_ids: Tensor,               # (B, T)
        targets: Optional[Tensor] = None,
        state: Optional[dict] = None,
        return_audit: bool = False,
        lambda_err: float = 0.1,
        lambda_err_mid: Optional[float] = None,
        lambda_err_coarse: Optional[float] = None,
        lambda_bal: float = 0.01,
        write_slow: bool = True,
        k_depth: Optional[int] = None,
        collect_stats: bool = True,
        label_smoothing: float = 0.0,
    ) -> ModelOutput:
        """
        Looped form only (token-by-token outer loop removed).

        1. Embed + dendrite windows over all T
        2. for j in 1..k_depth (shared weights):
             predict / gate / router over T
             fast-memory causal scan via write_seq (rank-1 recurrence, not an outer token API)
             neural field over T
        3. Final router + slow ring + logits

        Decode still passes T=1 with state; that is the same function, not a second code path.
        """
        B, T = input_ids.shape
        device = input_ids.device
        ab = self.ablation
        # per-position gate/memory norms cost O(T*k) tiny kernels; only when needed
        collect = bool(collect_stats or return_audit)
        if state is None:
            state = self._init_state(B, device)
        if T == 0:
            empty_logits = input_ids.new_zeros(B, 0, self.cfg.vocab_size).float()
            return ModelOutput(
                logits=empty_logits,
                loss=None,
                metrics={"path": "parallel", "k_depth": 0.0},
                audit=None,
                state=state,
            )

        # Metacognition: set depth k + encoding bias δ from *previous* step signals
        # (carried in state). Same path for train (T>1) and decode (T=1).
        self.meta.training = bool(self.training)
        gate_bias_meta = None  # (B,) or None
        want_replay = bool(state.get("want_replay") or False)

        # Long-session resilience: periodic / opportunistic self-heal
        # - always heal if prior health flagged bad
        # - or every ~64 tokens of segment progress
        _seg = int(state.get("seg_pos") or 0)
        _heal_every = int(self.cfg.heal_every_tokens)
        _need_heal = bool(state.get("_force_heal")) or (_heal_every > 0 and _seg > 0 and _seg % _heal_every == 0)
        if _need_heal or state.get("_memory_health", {}).get("fast_finite", 1.0) < 0.99:
            state = self.heal_session_state(state) or state
        bar_S_state = state.get("bar_S")
        # Always run meta for encoding bias δ when enabled; adaptive *k* only on
        # decode steps (T==1). Multi-token train/prefill keeps k_max so one
        # parallel depth applies to the whole chunk (logits stay prefix-consistent).
        last_S = state.get("last_S")
        if last_S is None or not torch.is_tensor(last_S):
            last_S = torch.zeros(B, device=device)
        else:
            last_S = last_S.to(device).reshape(-1)
            if last_S.numel() == 1 and B > 1:
                last_S = last_S.expand(B)
            elif last_S.numel() != B:
                last_S = last_S.mean().expand(B)

        if ab.use_metacognition:
            ent_prev = state.get("last_router_ent")
            mem_n = state.get("M")
            mem_norm = mem_n.detach().norm(dim=(-2, -1)) if isinstance(mem_n, Tensor) else None
            # Previous field state (if any) enriches J-space + consciousness
            h_prev = state.get("H")
            if h_prev is not None and torch.is_tensor(h_prev):
                h_for_meta = h_prev.reshape(-1, h_prev.size(-1))[:B] if h_prev.ndim >= 2 else h_prev
            else:
                h_for_meta = None
            decision = self.meta.step(
                last_S,
                bar_S=bar_S_state if torch.is_tensor(bar_S_state) else None,
                router_entropy=ent_prev if torch.is_tensor(ent_prev) else None,
                mem_norm=mem_norm,
                n_tokens=T,
                h=h_for_meta,
            )
            # Keep workspace for field / router injection this step
            self._last_jspace_broadcast = getattr(decision, "jspace_broadcast", None)
            self._last_consciousness = getattr(decision, "consciousness", None)
            # Consciousness ignition → slightly stronger memory encoding
            if self._last_consciousness is not None and self._last_consciousness.ignition:
                gate_boost = float(self.cfg.consciousness_gate_boost)
                decision.gate_bias = (decision.gate_bias + gate_boost).clamp(-1.0, 1.0)
            # Deliberate mode can request extra depth
            if self._last_consciousness is not None and self._last_consciousness.mode == "deliberate":
                k_from_c = max(int(decision.k), min(int(self.cfg.k_max), int(decision.k) + 1))
                decision.k = k_from_c
            gate_bias_meta = decision.gate_bias  # (B,)
            bar_S_state = decision.bar_S
            want_replay = bool(decision.want_replay)
            k_from_meta = max(1, min(int(self.cfg.k_max), int(decision.k)))
        else:
            k_from_meta = max(1, int(ab.fixed_k))

        if k_depth is not None:
            k_depth = max(1, int(k_depth))
        elif not ab.use_metacognition:
            k_depth = max(1, int(ab.fixed_k))
        elif T == 1 and bool(getattr(self.cfg, "adaptive_decode_depth", False)):
            k_depth = k_from_meta  # opt-in adaptive effort (train/decode depth mismatch + carry reset)
        else:
            # train / prefill / decode: same depth k_max, so per-pass memory carry is never dropped
            k_depth = max(1, int(self.cfg.k_max))

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
        ent_list: list = []
        write_norms: list[Tensor] = []
        mem_norms: list[Tensor] = []
        pi: Tensor | None = None

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
        # running surprise sum per pass (so the relative-surprise baseline is identical for one-call,
        # chunked and token-by-token decoding); only valid when the pass carry itself is valid
        _sc = state.get("S_cum")
        Scum_prev = _sc if (carried and isinstance(_sc, list) and len(_sc) == k_depth) else [None] * k_depth
        n_seen_prev = int(state.get("S_n") or 0) if Scum_prev[0] is not None else 0
        Scum_new: list[Tensor] = []

        # slow-ring write cadence must match decoding: one snapshot per `segment_len` tokens (counted across calls)
        seg_cnt = int(state.get("seg_pos") or 0)
        snaps: list[tuple] = []  # (ti, M, S)
        # Frozen sessions: g=0 already; also skip slow-ring snapshots (paper §7)
        will_write_slow = bool(
            write_slow and ab.use_slow_memory and ab.use_fast_memory and not frozen
        )

        # Learned workspace: read -> compete -> broadcast INSIDE every loop (per token, causal).
        # Legacy workspace: one broadcast from the previous call, projected once per forward.
        _ws_learned = self.workspace is not None
        _use_ws = _ws_learned and bool(getattr(ab, "use_workspace", True))
        _ws_code: Optional[Tensor] = None
        _lam_lens = float(getattr(self.cfg, "lambda_lens", 0.0) or 0.0)
        _lens_on = _use_ws and self.training and targets is not None and _lam_lens > 0.0
        lens_loss = X.new_zeros(())
        _js = None if _ws_learned else getattr(self, "_last_jspace_broadcast", None)
        if _js is not None and torch.is_tensor(_js) and getattr(self.field, "W_js", None) is not None:
            if _js.ndim == 1:
                _js_bt = _js.unsqueeze(0).expand(B * T, -1)
            elif _js.ndim == 2 and _js.size(0) == B:
                _js_bt = _js.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            else:
                _js_bt = _js.reshape(-1, _js.size(-1))[: B * T]
                if _js_bt.size(0) < B * T:
                    _js_bt = _js_bt[0:1].expand(B * T, -1)
            _js_proj = self.field.project_js(_js_bt, B * T)
        else:
            _js_proj = None

        for _j in range(k_depth):
            if _use_ws:
                _ws_b, _ws_code = self.workspace(H.reshape(B * T, -1), steer=self._ws_steer)
                _js_proj_j = self.field.project_js(_ws_b, B * T)
                js_r_j = _ws_b
                if _lens_on and _j == (k_depth - 1) // 2:
                    lens_loss = self._lens_loss(H, targets)
            else:
                _js_proj_j = _js_proj
                js_r_j = None if _ws_learned else getattr(self, "_last_jspace_broadcast", None)
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
                activity = H.abs().mean(dim=-1)  # (B, T)
                # δ from causal relative surprise (cumsum) — no future tokens in the baseline.
                # Metacognition owns k (decode), want_replay, bar_S tracking separately.
                t_idx = torch.arange(1, T + 1, device=S.device, dtype=S.dtype).view(1, T) + float(n_seen_prev)
                cs = S.detach().cumsum(dim=-1)
                if Scum_prev[_j] is not None:
                    cs = cs + Scum_prev[_j].to(device=cs.device, dtype=cs.dtype)
                cmean = cs / t_idx
                Scum_new.append(cs[:, -1:].detach())
                rel = S / cmean.clamp(min=1e-4)
                delta = torch.zeros_like(S)
                delta = torch.where(rel > 1.5, torch.full_like(delta, 0.5), delta)
                delta = torch.where(rel < 0.5, torch.full_like(delta, -0.5), delta)
                # Train-only: add sequence-level meta encoding bias (control-path diagram).
                # Not applied in eval so full-sequence ↔ token-by-token logits stay matched.
                if gate_bias_meta is not None and self.training:
                    delta = delta + gate_bias_meta.to(dtype=S.dtype, device=S.device).view(B, 1)
                logit = (
                    self.neuromod.a_s * (S - 0.5)
                    + self.neuromod.a_c * activity
                    + self.neuromod.bias
                    + delta
                )
                G = torch.sigmoid(logit).unsqueeze(-1).clamp(max=self.safety.gate_cap)
            else:
                G = torch.full(
                    (B, T, 1), float(ab.fixed_gate), device=device, dtype=X.dtype
                ).clamp(max=self.safety.gate_cap)
            # Train-time floor stops gate collapse (writes stay informative)
            g_floor = float(getattr(self.safety, "gate_floor", 0.05) or 0.0)
            if self.training and g_floor > 0.0 and not frozen and ab.use_neuromodulator:
                G = G * (1.0 - g_floor) + g_floor
            if frozen:
                G = torch.zeros_like(G)

            D = D_win + self.dendrite.gamma * E

            # Token-parallel router (positions independent given H)
            o_flat, pi_flat, _w = self.router(H.reshape(B * T, -1), workspace=js_r_j)
            O = o_flat.view(B, T, -1)
            pi = pi_flat.view(B, T, -1)
            if collect:
                ent_list = list((-(pi * pi.clamp_min(1e-12).log()).sum(-1).mean(0)).detach())
            bal_loss = bal_loss + (pi.mean(0) - 1.0 / max(1, self.cfg.n_skills)).pow(2).sum()

            if ab.use_fast_memory:
                # Every depth pass owns its fast memory. Carry across decode calls.
                M = (
                    M_prev[_j]
                    if M_prev[_j] is not None
                    else torch.zeros(B, self.cfg.d_k, self.cfg.d_v, device=device, dtype=X.dtype)
                )
                d_prev0 = D_prev_pass[_j] if D_prev_pass[_j] is not None else None
                M, R, more_snaps, seg_cnt, w_n, m_n = self.fast_mem.write_seq(
                    M, D, O, G,
                    d_prev0=d_prev0,
                    truncate_window=self._truncate_window,
                    max_memory_norm=float(self.safety.max_memory_norm or 0),
                    segment_len=self._segment_len if (will_write_slow and _j == k_depth - 1) else 0,
                    seg_cnt0=seg_cnt,
                    surprises=S if (will_write_slow and _j == k_depth - 1) else None,
                    collect_stats=collect,
                )
                snaps.extend(more_snaps)
                if collect:
                    gates_audit.extend([G[:, t].detach().mean() for t in range(T)])
                    write_norms.extend(w_n)
                    mem_norms.extend(m_n)
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
            H = self.field.single_step(
                H.reshape(B * T, -1),
                D.reshape(B * T, -1),
                R_shift.reshape(B * T, -1),
                js=None,
                loop_idx=_j,
                total_k=k_depth,
                js_proj=_js_proj_j,
            ).view(B, T, -1)

        # Final router on refined H — this O feeds logits; keep its π for
        # entropy regularisation / metacognition (was using last depth-pass π).
        if _use_ws:
            js_r, _ws_code = self.workspace(H.reshape(B * T, -1), steer=self._ws_steer)
        else:
            js_r = None if _ws_learned else getattr(self, "_last_jspace_broadcast", None)
        if _ws_code is not None:
            self._last_ws_code = _ws_code.detach()
        o_flat, pi_flat, _w = self.router(H.reshape(B * T, -1), workspace=js_r)
        O = o_flat.view(B, T, -1)
        pi = pi_flat.view(B, T, -1)

        if ab.use_slow_memory:
            slow_slots = state.get("slow_slots")
            slow_filled = state.get("slow_filled")
            slow_ptr = int(state.get("slow_ptr") or 0)
            _sk = state.get("slow_slot_keys")
            if _sk is not None and (will_write_slow and snaps):
                _sk = None
            # Causal slow read + ring writes (must match token-by-token decode).
            # U[t] only sees segment snapshots completed at positions < t.
            # No end-of-call flush when T>1 (that broke prefix ↔ decode parity).
            if will_write_slow and T > 0 and snaps:
                # Between snap boundaries the ring is constant → batch-read those
                # spans with read_seq instead of a Python per-token loop.
                U_parts: list[Tensor] = []
                snap_i = 0
                t0 = 0
                while t0 < T:
                    # advance snaps that close exactly at positions we already passed
                    t1 = T
                    if snap_i < len(snaps):
                        t1 = min(T, snaps[snap_i][0] + 1)  # read through the snap position
                    # read [t0, t1) with current ring (snap at t is visible only from t+1)
                    if t1 > t0:
                        U_parts.append(
                            self.slow_mem.read_seq(
                                D[:, t0:t1], slow_slots, slow_filled, slot_keys=_sk
                            )
                        )
                    # apply snaps that close at t1-1 (last position just read)
                    while snap_i < len(snaps) and snaps[snap_i][0] < t1:
                        _ti, M_snap, s_snap = snaps[snap_i]
                        slow_slots, slow_ptr, slow_filled = self.slow_mem.write_slot(
                            slow_slots, slow_ptr, slow_filled, M_snap,
                            weight=s_snap.clamp(0.5, 2.0),
                        )
                        snap_i += 1
                    t0 = t1
                U = torch.cat(U_parts, dim=1) if U_parts else D.new_zeros(B, T, self.cfg.d_v)
                seg_pos = seg_cnt
            else:
                U = self.slow_mem.read_seq(D, slow_slots, slow_filled, slot_keys=_sk)
                if will_write_slow and T > 0:
                    # no snaps this call; just carry segment counter
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
            self.out_norm(
                O + self.mem_scale * self.P_f(R) + self.slow_scale * self.P_s(U)
            )
        )
        if bool(getattr(self.cfg, "scale_tied_logits", False)):
            logits = logits * (float(self.cfg.d_model) ** -0.5)

        loss = None
        # Host metrics only when collecting (log / audit) — avoid per-step GPU sync
        metrics: dict[str, float] = {
            "path": "parallel",
            "k_depth": float(k_depth),
            "want_replay": 1.0 if want_replay else 0.0,
            "segment_len": float(self._segment_len),
        }
        if collect and _use_ws and _ws_code is not None:
            metrics.update(GlobalWorkspace.stats(_ws_code))
        if collect:
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
            metrics.update(
                {
                    "gate_mean": _gate_mean,
                    "mem_norm": _mem_mean,
                    "mean_S": float(S.detach().mean().item()) if T > 0 else 0.0,
                    "memory_healed": 1.0 if state.get("_memory_health") else 0.0,
                    "meta_active": 1.0 if (ab.use_metacognition and gate_bias_meta is not None) else 0.0,
                    "slow_filled": float(
                        slow_filled.detach().float().sum().item()
                    ) if isinstance(slow_filled, Tensor) else 0.0,
                }
            )
            if gate_bias_meta is not None:
                metrics["gate_bias_mean"] = float(gate_bias_meta.detach().float().mean().item())
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                label_smoothing=float(max(0.0, min(0.5, label_smoothing))),
            )
            # Keep gate and skill mix alive so memory/router actually contribute
            g_mean = G.mean()
            gate_target = float(getattr(self.safety, "gate_target", 0.35) or 0.35)
            lambda_gate = float(getattr(self.safety, "lambda_gate", 0.01) or 0.01)
            gate_reg = (g_mean - gate_target).pow(2)
            # mild entropy floor on last-pass router (avoid single-skill collapse)
            if isinstance(pi, Tensor):
                ent = -(pi * pi.clamp_min(1e-12).log()).sum(-1).mean()
                n_sk = max(2, int(self.cfg.n_skills))
                ent_max = float(__import__("math").log(n_sk))
                ent_reg = F.relu(0.2 * ent_max - ent)  # penalize very low entropy
            else:
                ent_reg = g_mean.new_zeros(())
            lambda_ent = float(getattr(self.safety, "lambda_ent", 0.005) or 0.005)
            # Fine error from depth loop; mid/coarse = hierarchical levels
            err_fine = err_energy / max(1, k_depth)
            err_mid = logits.new_zeros(())
            err_coarse = logits.new_zeros(())
            lam_m = float(
                lambda_err_mid
                if lambda_err_mid is not None
                else (getattr(self.cfg, "lambda_err_mid", 0.05) or 0.0)
            )
            lam_c = float(
                lambda_err_coarse
                if lambda_err_coarse is not None
                else (getattr(self.cfg, "lambda_err_coarse", 0.02) or 0.0)
            )
            if (
                self.training
                and T > 1
                and hasattr(self.predictor, "level_errors")
                and (lam_m > 0.0 or lam_c > 0.0)
            ):
                le = self.predictor.level_errors(
                    H, X, need_fine=False, need_mid=lam_m > 0.0, need_coarse=lam_c > 0.0
                )
                err_mid = le["err_mid"]
                err_coarse = le["err_coarse"]
            loss = (
                ce
                + lambda_err * err_fine
                + lam_m * err_mid
                + lam_c * err_coarse
                + lambda_bal * (bal_loss / max(1, T * k_depth))
                + lambda_gate * gate_reg
                + lambda_ent * ent_reg
                + _lam_lens * lens_loss
            )
            # Host sync only when logging (collect_stats) — speeds up train step
            if collect:
                metrics.update(
                    {
                        "ce": float(ce.detach().item()),
                        "err_energy": float(err_fine.detach().item()),
                        "err_mid": float(err_mid.detach().item()) if torch.is_tensor(err_mid) else 0.0,
                        "err_coarse": float(err_coarse.detach().item()) if torch.is_tensor(err_coarse) else 0.0,
                        "gate_reg": float(gate_reg.detach().item()),
                        "ent_reg": float(ent_reg.detach().item()) if torch.is_tensor(ent_reg) else 0.0,
                        "lens_loss": float(lens_loss.detach().item()) if _lens_on else 0.0,
                        "lambda_err": float(lambda_err),
                        "lambda_err_mid": float(lam_m),
                        "lambda_err_coarse": float(lam_c),
                    }
                )

        audit = None
        if return_audit:
            audit = {
                "path": "parallel",
                "gates": [float(g) for g in gates_audit[-T:]] if gates_audit else (
                    [float(G.detach().mean().item())] if T > 0 else []
                ),
                "write_norms": [float(x) for x in write_norms[-T:]] if write_norms else [],
                "mem_norms": [float(x) for x in mem_norms[-T:]] if mem_norms else (
                    [float(M.detach().norm().item())] if isinstance(M, Tensor) else []
                ),
                "ks": [float(k_depth)] * max(1, T),
                "router_entropy": [float(e) for e in ent_list] if ent_list else [],
                "surprises": [float(s) for s in S.detach().mean(0)] if T > 0 else [],
                "mean_S": float(S.detach().mean().item()) if T > 0 else 0.0,
                "final_mem_norm": float(M.detach().norm()) if isinstance(M, Tensor) else 0.0,
                "memory_frozen": bool(frozen),
            }

        # Signals for the *next* call's metacognition (decode T=1 or next train batch)
        last_S_out = S.detach().mean(dim=-1) if T > 0 else torch.zeros(B, device=device)
        # mean router entropy over positions (last pass)
        if T > 0 and isinstance(pi, Tensor):
            last_ent_out = (-(pi * pi.clamp_min(1e-12).log()).sum(-1)).mean(dim=-1).detach()
        else:
            last_ent_out = state.get("last_router_ent")

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
            "S_cum": Scum_new,
            "S_n": n_seen_prev + T,
            "slow_slots": slow_slots,
            "slow_filled": slow_filled,
            "slow_ptr": slow_ptr,
            "seg_pos": seg_pos,
            "memory_frozen": frozen,
            "last_S": last_S_out,
            "last_router_ent": last_ent_out,
            "bar_S": bar_S_state.detach() if torch.is_tensor(bar_S_state) else bar_S_state,
            "want_replay": want_replay,
        }
        return ModelOutput(
            logits=logits, loss=loss, metrics=metrics, audit=audit, state=new_state
        )



    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_id: Optional[int] = None,
        state: Optional[dict] = None,
        heal_every: Optional[int] = None,
    ) -> Tensor:
        """Autoregressive generate with SENTINEL InferenceCache (KV-cache analogue).

        Prefill runs once on the prompt; each new token is T=1 decode with
        carried fixed-size memory — no growing attention KV.
        """
        from sentinel.modules.inference_cache import InferenceCache
        self.eval()
        if heal_every is None:
            heal_every = int(self.cfg.heal_every_tokens)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B = input_ids.size(0)
        device = input_ids.device
        cache = InferenceCache.from_model(self, batch=B, device=device)
        if state is not None:
            cache.state = state
        return cache.generate(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
            heal_every=heal_every,
        )

    def heal_session_state(self, state: Optional[dict]) -> Optional[dict]:
        """Long-session + corrupt-checkpoint resilience.

        Sanitizes fast M and slow ring in-place (returns new dict).
        Safe to call every turn or periodically during long inference.
        """
        if not state:
            return state
        try:
            from sentinel.modules.memory import heal_matrix, heal_slots, memory_health_report
        except ImportError:
            from .modules.memory import heal_matrix, heal_slots, memory_health_report

        M = state.get("M")
        slots = state.get("slow_slots")
        filled = state.get("slow_filled")
        h = state.get("h")
        r = state.get("r")
        error = state.get("error")

        if M is not None and torch.is_tensor(M):
            M = heal_matrix(M, max_norm=float(self.safety.max_memory_norm))
        if slots is not None and filled is not None:
            slots, filled = heal_slots(slots, filled, max_norm=float(self.safety.max_memory_norm))
        if h is not None and torch.is_tensor(h):
            h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
            hn = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h = h * (50.0 / hn).clamp(max=1.0)
        if r is not None and torch.is_tensor(r):
            r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        if error is not None and torch.is_tensor(error):
            error = torch.where(torch.isfinite(error), error, torch.zeros_like(error))

        out = {
            **state,
            "M": M if M is not None else state.get("M"),
            "slow_slots": slots if slots is not None else state.get("slow_slots"),
            "slow_filled": filled if filled is not None else state.get("slow_filled"),
            "h": h if h is not None else state.get("h"),
            "r": r if r is not None else state.get("r"),
            "error": error if error is not None else state.get("error"),
        }
        out["_memory_health"] = memory_health_report(M, slots, filled)
        return out

    def consolidate_turn(self, state: Optional[dict], weight: float = 1.0) -> Optional[dict]:
        """End-of-turn consolidation (paper v3, Section 11): snapshot the final fast memory M into the slow ring.

        forward() writes the ring only at global segment boundaries (so a training call == prefill + decode). A short chat
        turn may cross no boundary and would leave the user's slow memory empty, so the per-user memory calls this once per
        turn. Returns a new state dict (the input is not modified).
        """
        ab = self.ablation
        if not state or not (ab.use_slow_memory and ab.use_fast_memory) or state.get("slow_slots") is None:
            return state
        M = state.get("M")
        if M is None:
            return state
        w = torch.full((M.size(0),), float(weight), device=M.device)
        slots, ptr, filled = self.slow_mem.write_slot(
            state["slow_slots"], int(state.get("slow_ptr") or 0), state["slow_filled"], M, weight=w
        )
        out = {**state, "slow_slots": slots, "slow_ptr": ptr, "slow_filled": filled}
        return self.heal_session_state(out) or out

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

# Backward-compatible aliases
NeuroField = SENTINEL

