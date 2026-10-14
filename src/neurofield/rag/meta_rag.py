"""Metacognition-controlled Adaptive RAG.

Puts retrieval under the same monitor→control loop as the rest of NeuroField:

  MONITOR  — confidence / arousal / conflict (from Metacognition or a prompt heuristic)
  CONTROL  — should_retrieve?, top_k, min_score

Design goals
------------
* Low confidence or high arousal  → retrieve (more chunks)
* High confidence                 → skip RAG (faster, cheaper, no noise)
* Optional real meta signals from the model; falls back to a prompt heuristic
  so the path works even before the first forward pass.
* Hysteresis + cache so we do not thrash on borderline scores.
* Latency-safe: at most one retrieve per chat turn by default.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from neurofield.rag.pipeline import RAGPipeline, get_default_rag


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class MetaRAGDecision:
    """What the metacognitive controller decided for this turn."""

    should_retrieve: bool
    top_k: int
    min_score: float
    confidence: float          # 0..1  (higher = more sure, less need for external knowledge)
    arousal: float             # non-negative urgency
    conflict: float            # uncertainty / ambiguity
    reason: str
    source: str                # "meta_signals" | "prompt_heuristic" | "forced"
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt-level heuristic (used when real meta signals are not yet available)
# ---------------------------------------------------------------------------

_QUESTION_RE = re.compile(
    r"\b(what|why|how|when|where|who|which|explain|describe|define|tell me|"
    r"kya|kyun|kaise|kab|kahan|kaun|batao|samjhao|detail)\b",
    re.I,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(maybe|perhaps|not sure|i think|possibly|might|could be|"
    r"shayad|lagta|pata nahi|doubt)\b",
    re.I,
)
_FACTUAL_RE = re.compile(
    r"\b(who is|what is|when did|capital of|population|founded|definition|"
    r"according to|paper|paper says|document|policy|api|endpoint|error code)\b",
    re.I,
)
_OPINION_RE = re.compile(
    r"\b(opinion|feel|prefer|best|worst|should i|recommend)\b",
    re.I,
)


def estimate_prompt_confidence(prompt: str) -> tuple[float, float, float, str]:
    """
    Cheap zero-model estimate of (confidence, arousal, conflict, reason).

    Returns values in roughly the same ranges Metacognition produces so the
    same thresholds can be used.
    """
    text = (prompt or "").strip()
    if not text:
        return 0.4, 0.6, 0.5, "empty_prompt"

    n_tokens = max(1, len(text.split()))
    conf = 0.55
    arousal = 0.3
    conflict = 0.2
    reasons: list[str] = []

    if _QUESTION_RE.search(text):
        conf -= 0.12
        arousal += 0.15
        reasons.append("question")
    if _FACTUAL_RE.search(text):
        conf -= 0.18
        arousal += 0.25
        conflict += 0.1
        reasons.append("factual")
    if _UNCERTAINTY_RE.search(text):
        conf -= 0.2
        conflict += 0.25
        reasons.append("uncertainty_markers")
    if _OPINION_RE.search(text):
        conf += 0.1  # opinions need less external retrieval
        reasons.append("opinion")
    if n_tokens < 6:
        conf -= 0.08
        conflict += 0.1
        reasons.append("short")
    if n_tokens > 40:
        conf -= 0.05
        arousal += 0.1
        reasons.append("long")

    # Clamp
    conf = max(0.05, min(0.95, conf))
    arousal = max(0.0, min(3.0, arousal))
    conflict = max(0.0, min(2.0, conflict))
    reason = "+".join(reasons) if reasons else "neutral"
    return conf, arousal, conflict, reason


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MetaRAGController:
    """
    Metacognition → RAG control layer (Auto RAG).

    Modes
    -----
    * ``meta``  — retrieve only when confidence is low / arousal-conflict high
    * ``auto``  — always try retrieve for informational prompts; meta still
                  sets dynamic top_k. Default for production "Auto RAG".
    * ``always``— force retrieve every turn
    * ``off``   — never retrieve

    Thresholds are mild by default; tune via constructor or env.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.58,
        arousal_force: float = 1.1,
        conflict_force: float = 1.2,
        base_top_k: int = 4,
        min_top_k: int = 2,
        max_top_k: int = 8,
        min_score: float = 0.12,
        enabled: bool = True,
        mode: str = "auto",
        rag: Optional[RAGPipeline] = None,
    ):
        self.confidence_threshold = float(
            os.environ.get("NEUROFIELD_META_RAG_CONF", confidence_threshold)
        )
        self.arousal_force = float(arousal_force)
        self.conflict_force = float(conflict_force)
        self.base_top_k = int(base_top_k)
        self.min_top_k = int(min_top_k)
        self.max_top_k = int(max_top_k)
        self.min_score = float(min_score)
        env_enabled = os.environ.get("NEUROFIELD_META_RAG")
        self.enabled = bool(
            enabled
            if env_enabled is None
            else env_enabled not in ("0", "false", "False")
        )
        env_mode = os.environ.get("NEUROFIELD_RAG_MODE", mode).strip().lower()
        if env_mode not in ("auto", "meta", "always", "off"):
            env_mode = "auto"
        self.mode = env_mode
        self._rag = rag
        # Simple hysteresis: last decision per session key
        self._last: dict[str, bool] = {}

    @property
    def rag(self) -> RAGPipeline:
        if self._rag is None:
            self._rag = get_default_rag()
        return self._rag

    def decide(
        self,
        prompt: str,
        *,
        confidence: Optional[float] = None,
        arousal: Optional[float] = None,
        conflict: Optional[float] = None,
        force: Optional[bool] = None,
        session_key: str = "default",
    ) -> MetaRAGDecision:
        """
        Produce a retrieval decision.

        Priority:
          1. force=True/False (explicit override)
          2. real metacognition signals if provided
          3. prompt heuristic
        """
        if (not self.enabled or self.mode == "off") and force is not True:
            return MetaRAGDecision(
                should_retrieve=False,
                top_k=self.base_top_k,
                min_score=self.min_score,
                confidence=1.0,
                arousal=0.0,
                conflict=0.0,
                reason="meta_rag_disabled" if not self.enabled else "mode_off",
                source="forced",
            )

        if force is True or self.mode == "always":
            return MetaRAGDecision(
                should_retrieve=True,
                top_k=self.base_top_k,
                min_score=self.min_score,
                confidence=0.0,
                arousal=2.0,
                conflict=1.0,
                reason="forced_on" if force is True else "mode_always",
                source="forced",
            )
        if force is False:
            return MetaRAGDecision(
                should_retrieve=False,
                top_k=self.base_top_k,
                min_score=self.min_score,
                confidence=1.0,
                arousal=0.0,
                conflict=0.0,
                reason="forced_off",
                source="forced",
            )

        source = "meta_signals"
        reason_bits: list[str] = [f"mode={self.mode}"]

        if confidence is None or arousal is None or conflict is None:
            c, a, conf_lict, heur_reason = estimate_prompt_confidence(prompt)
            confidence = confidence if confidence is not None else c
            arousal = arousal if arousal is not None else a
            conflict = conflict if conflict is not None else conf_lict
            source = "prompt_heuristic"
            reason_bits.append(heur_reason)
        else:
            reason_bits.append("live_meta")

        confidence = float(confidence)
        arousal = float(arousal)
        conflict = float(conflict)

        # Core meta rule: retrieve when unsure or highly aroused/conflicted
        need = (
            confidence < self.confidence_threshold
            or arousal >= self.arousal_force
            or conflict >= self.conflict_force
        )

        # Auto mode: also retrieve for informational / question-like prompts
        # even when confidence is moderate — still skips pure chit-chat.
        if self.mode == "auto" and not need:
            informational = (
                confidence < 0.75
                or arousal >= 0.4
                or conflict >= 0.35
                or bool(_QUESTION_RE.search(prompt or ""))
                or bool(_FACTUAL_RE.search(prompt or ""))
            )
            if informational and len(self.rag.store) > 0:
                need = True
                reason_bits.append("auto_informational")

        # Hysteresis: if last turn retrieved and we are only mildly confident, keep retrieving
        last = self._last.get(session_key)
        if last is True and confidence < (self.confidence_threshold + 0.12):
            need = True
            reason_bits.append("hysteresis")

        # Dynamic top_k: more uncertainty → more chunks
        uncertainty = (1.0 - confidence) + 0.35 * min(conflict, 2.0) + 0.2 * min(arousal, 2.0)
        frac = max(0.0, min(1.0, uncertainty / 1.6))
        top_k = self.min_top_k + int(round((self.max_top_k - self.min_top_k) * frac))
        top_k = max(self.min_top_k, min(self.max_top_k, top_k))

        if need:
            reason_bits.append(
                f"conf={confidence:.2f}<{self.confidence_threshold}"
                if confidence < self.confidence_threshold
                else "retrieve"
            )
        else:
            reason_bits.append(f"conf={confidence:.2f} skip")

        decision = MetaRAGDecision(
            should_retrieve=need,
            top_k=top_k,
            min_score=self.min_score,
            confidence=confidence,
            arousal=arousal,
            conflict=conflict,
            reason="|".join(reason_bits),
            source=source,
            diagnostics={
                "uncertainty": round(uncertainty, 3),
                "threshold": self.confidence_threshold,
                "mode": self.mode,
                "store_size": len(self.rag.store),
            },
        )
        self._last[session_key] = decision.should_retrieve
        return decision

    def maybe_augment(
        self,
        prompt: str,
        *,
        confidence: Optional[float] = None,
        arousal: Optional[float] = None,
        conflict: Optional[float] = None,
        force: Optional[bool] = None,
        session_key: str = "default",
    ) -> tuple[str, MetaRAGDecision, list[dict]]:
        """
        Decide + (optionally) retrieve + return augmented prompt.

        Returns:
          (final_prompt, decision, hits)
        """
        decision = self.decide(
            prompt,
            confidence=confidence,
            arousal=arousal,
            conflict=conflict,
            force=force,
            session_key=session_key,
        )
        if not decision.should_retrieve:
            return prompt, decision, []

        hits = self.rag.retrieve(prompt, top_k=decision.top_k)
        # Filter by min_score if the store provides scores
        hits = [h for h in hits if float(h.get("score") or 0.0) >= decision.min_score]
        if not hits:
            decision.reason += "|no_hits_above_min_score"
            return prompt, decision, []

        block_lines = []
        for i, h in enumerate(hits, 1):
            src = h.get("source") or "kb"
            block_lines.append(f"[{i}] ({src}, score={h.get('score', 0):.3f}) {h['text']}")
        block = "\n".join(block_lines)
        augmented = (
            "[Retrieved context — prefer these facts over guessing; "
            "metacognition requested external knowledge]\n"
            f"{block}\n\n"
            f"User question: {prompt}"
        )
        return augmented, decision, hits


# Process-wide default (same pattern as get_default_rag)
_DEFAULT_CONTROLLER: Optional[MetaRAGController] = None


def get_meta_rag(reload: bool = False) -> MetaRAGController:
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is not None and not reload:
        return _DEFAULT_CONTROLLER
    _DEFAULT_CONTROLLER = MetaRAGController(mode="auto")
    return _DEFAULT_CONTROLLER


def auto_rag(
    prompt: str,
    *,
    confidence: Optional[float] = None,
    arousal: Optional[float] = None,
    conflict: Optional[float] = None,
    force: Optional[bool] = None,
    session_key: str = "default",
    top_k: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """
    One-call Auto RAG entrypoint for chat, agent, scripts.

    Returns
    -------
    (final_prompt, info)
        final_prompt is either the original or the context-augmented string.
        info has should_retrieve, top_k, confidence, hits, reason, mode, …
    """
    ctrl = get_meta_rag()
    if top_k is not None:
        # temporary override without mutating global defaults long-term
        old_min, old_max, old_base = ctrl.min_top_k, ctrl.max_top_k, ctrl.base_top_k
        ctrl.min_top_k = ctrl.max_top_k = ctrl.base_top_k = int(top_k)
    try:
        augmented, decision, hits = ctrl.maybe_augment(
            prompt,
            confidence=confidence,
            arousal=arousal,
            conflict=conflict,
            force=force,
            session_key=session_key,
        )
    finally:
        if top_k is not None:
            ctrl.min_top_k, ctrl.max_top_k, ctrl.base_top_k = old_min, old_max, old_base

    info: dict[str, Any] = {
        "should_retrieve": decision.should_retrieve,
        "top_k": decision.top_k,
        "confidence": round(decision.confidence, 3),
        "arousal": round(decision.arousal, 3),
        "conflict": round(decision.conflict, 3),
        "reason": decision.reason,
        "source": decision.source,
        "mode": decision.diagnostics.get("mode", ctrl.mode),
        "hits": len(hits),
        "hit_scores": [round(float(h.get("score") or 0), 3) for h in hits[:6]],
        "store_size": len(ctrl.rag.store),
    }
    return augmented, info
