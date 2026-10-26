"""
Consciousness Simulator for SENTINEL.

Lightweight, architectural simulator of *access consciousness* inspired by:

  • Global Workspace Theory (Baars / Dehaene) — ignition + broadcast
  • Anthropic J-space (reportable, controllable internal workspace)
  • Higher-order / metacognitive monitoring (already in Metacognition)
  • A cheap Integrated-Information *proxy* (not full IIT Φ, which is intractable)

This does **not** claim the model is phenomenally conscious.
It implements measurable functional correlates used in GWT-style models:

  - Limited-capacity workspace (J-space)
  - Non-linear ignition when signals cross a threshold
  - Global broadcast availability
  - Reportability / verbalizability
  - Distinction between routine (System-1) and deliberate (System-2) processing
  - A simple integration proxy from occupancy + field coherence

Use cases:
  - Logging / interpretability ("what is currently in the workspace?")
  - Steering experiments (inject → observe ignition / report)
  - Research ablations of conscious-access mechanisms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class ConsciousnessState:
    """Snapshot of simulated access-consciousness at one step."""

    # Overall access level in [0, 1]
    level: float

    # GWT-style ignition (non-linear amplification into workspace)
    ignition: bool

    # Broadcast strength (how strongly content is available to the rest of the net)
    broadcast_strength: float

    # Cheap integration proxy (0..1); higher ≈ more unified workspace
    phi_proxy: float

    # Whether content is considered reportable / verbalizable
    reportable: bool

    # High-level mode
    mode: str  # "unconscious" | "access" | "deliberate"

    # Optional human-readable summary
    summary: str = ""

    # Raw diagnostics
    diagnostics: dict[str, float] = field(default_factory=dict)


class ConsciousnessSimulator:
    """
    Functional consciousness-access simulator.

    Takes signals already produced by Metacognition + J-space (+ optional field state)
    and produces a structured ConsciousnessState.
    """

    def __init__(
        self,
        ignition_threshold: float = 0.55,
        access_threshold: float = 0.35,
        deliberate_threshold: float = 0.70,
        phi_occupancy_w: float = 0.45,
        phi_coherence_w: float = 0.35,
        phi_arousal_w: float = 0.20,
    ):
        self.ignition_threshold = float(ignition_threshold)
        self.access_threshold = float(access_threshold)
        self.deliberate_threshold = float(deliberate_threshold)
        self.phi_occupancy_w = float(phi_occupancy_w)
        self.phi_coherence_w = float(phi_coherence_w)
        self.phi_arousal_w = float(phi_arousal_w)

        # Running baseline for relative ignition (session-local)
        self._bar_drive: Optional[float] = None
        self._momentum = 0.92

    def step(
        self,
        *,
        jspace_capacity: float = 0.0,
        jspace_n_active: int = 0,
        jspace_reportable: bool = False,
        broadcast: Optional[Tensor] = None,
        arousal: Optional[Tensor] = None,
        conflict: Optional[Tensor] = None,
        confidence: Optional[Tensor] = None,
        process_mode: str = "system1",
        rel_S: Optional[Tensor] = None,
        h: Optional[Tensor] = None,
    ) -> ConsciousnessState:
        """
        Compute one ConsciousnessState from current control / workspace signals.
        """
        # --- scalar helpers ---
        def _mean(x: Optional[Tensor], default: float = 0.0) -> float:
            if x is None:
                return default
            if isinstance(x, (int, float)):
                return float(x)
            if not torch.is_tensor(x) or x.numel() == 0:
                return default
            return float(x.detach().float().mean())

        # Single host sync batch via stacked means when all tensor
        arousal_v = _mean(arousal, 0.0)
        conflict_v = _mean(conflict, 0.0)
        conf_v = _mean(confidence, 0.5)
        rel_s_v = _mean(rel_S, 0.0)

        # Drive into workspace (GWT "bottom-up + top-down")
        drive = 0.40 * arousal_v + 0.30 * conflict_v + 0.20 * jspace_capacity + 0.10 * max(0.0, rel_s_v - 0.5)
        drive = max(0.0, min(3.0, drive))

        # Running baseline → relative drive (helps ignition detection)
        if self._bar_drive is None:
            self._bar_drive = max(0.15, drive)
        else:
            self._bar_drive = self._momentum * self._bar_drive + (1.0 - self._momentum) * drive
        rel_drive = drive / (self._bar_drive + 1e-6)

        # --- Ignition (non-linear entry into global workspace) ---
        ignition = (
            rel_drive > self.ignition_threshold
            or (jspace_capacity > 0.5 and process_mode == "system2")
            or (jspace_n_active >= 3 and arousal_v > 0.8)
        )

        # --- Broadcast strength ---
        if broadcast is not None and broadcast.numel() > 0:
            b_norm = float(broadcast.detach().float().norm().item())
            # normalise roughly
            broadcast_strength = max(0.0, min(1.0, b_norm / (b_norm + 3.0) + 0.3 * jspace_capacity))
        else:
            broadcast_strength = max(0.0, min(1.0, 0.5 * jspace_capacity + 0.3 * float(ignition)))

        if ignition:
            broadcast_strength = min(1.0, broadcast_strength + 0.25)

        # --- Integration proxy (cheap stand-in for Φ) ---
        # Occupancy diversity + optional field coherence + arousal
        occ = max(0.0, min(1.0, jspace_capacity))
        coherence = 0.0
        if h is not None and h.numel() > 1:
            # simple coherence: 1 - normalised variance of h
            h_f = h.detach().float().reshape(-1)
            if h_f.numel() > 1:
                var = float(h_f.var().item())
                coherence = 1.0 / (1.0 + var)
        phi_proxy = (
            self.phi_occupancy_w * occ
            + self.phi_coherence_w * coherence
            + self.phi_arousal_w * min(1.0, arousal_v / 2.0)
        )
        phi_proxy = max(0.0, min(1.0, phi_proxy))
        if ignition:
            phi_proxy = min(1.0, phi_proxy + 0.1)

        # --- Overall access level ---
        level = (
            0.35 * broadcast_strength
            + 0.25 * phi_proxy
            + 0.20 * occ
            + 0.10 * (1.0 if ignition else 0.0)
            + 0.10 * (1.0 if process_mode == "system2" else 0.0)
        )
        level = max(0.0, min(1.0, level))

        # --- Mode ---
        if level >= self.deliberate_threshold or process_mode == "system2":
            mode = "deliberate"
        elif level >= self.access_threshold or ignition or jspace_reportable:
            mode = "access"
        else:
            mode = "unconscious"

        reportable = bool(jspace_reportable or (mode in ("access", "deliberate") and broadcast_strength > 0.25))

        # Summary line for logs
        summary = (
            f"mode={mode} level={level:.2f} ignition={ignition} "
            f"broadcast={broadcast_strength:.2f} φ≈{phi_proxy:.2f} "
            f"slots={jspace_n_active} reportable={reportable}"
        )

        diag = {
            "level": level,
            "ignition": 1.0 if ignition else 0.0,
            "broadcast_strength": broadcast_strength,
            "phi_proxy": phi_proxy,
            "jspace_capacity": float(jspace_capacity),
            "jspace_n_active": float(jspace_n_active),
            "arousal": arousal_v,
            "conflict": conflict_v,
            "confidence": conf_v,
            "rel_drive": float(rel_drive),
            "process_mode_s2": 1.0 if process_mode == "system2" else 0.0,
        }

        return ConsciousnessState(
            level=level,
            ignition=ignition,
            broadcast_strength=broadcast_strength,
            phi_proxy=phi_proxy,
            reportable=reportable,
            mode=mode,
            summary=summary,
            diagnostics=diag,
        )

    def reset(self) -> None:
        self._bar_drive = None
