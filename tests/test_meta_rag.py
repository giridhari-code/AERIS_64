"""Tests for metacognition-controlled Adaptive RAG."""

from __future__ import annotations

from neurofield.rag.meta_rag import (
    MetaRAGController,
    estimate_prompt_confidence,
    get_meta_rag,
)


def test_estimate_prompt_confidence_factual_question_is_uncertain():
    conf, arousal, conflict, reason = estimate_prompt_confidence(
        "What is the capital of India according to the document?"
    )
    assert conf < 0.55
    assert arousal > 0.3
    assert "factual" in reason or "question" in reason


def test_estimate_prompt_confidence_opinion_is_higher():
    conf, _, _, reason = estimate_prompt_confidence(
        "In your opinion, what is the best colour?"
    )
    assert conf > 0.4
    assert "opinion" in reason


def test_controller_skips_when_confident():
    # mode=meta so pure high-confidence chit-chat is skipped
    ctrl = MetaRAGController(confidence_threshold=0.5, enabled=True, mode="meta")
    d = ctrl.decide(
        "hello",
        confidence=0.9,
        arousal=0.1,
        conflict=0.1,
        session_key="t1",
    )
    assert d.should_retrieve is False
    assert d.source == "meta_signals"


def test_auto_mode_and_helper():
    from neurofield.rag.pipeline import RAGPipeline
    from neurofield.rag.meta_rag import auto_rag

    ctrl = MetaRAGController(mode="auto", enabled=True)
    ctrl._rag = RAGPipeline()
    ctrl._rag.ingest_text("NeuroField uses fast and slow memory.", source="test")
    d = ctrl.decide("What is NeuroField memory?", session_key="auto1")
    assert d.should_retrieve is True

    # auto_rag helper uses the process singleton; just ensure it returns a dict
    _aug, info = auto_rag("hello there", force=False, session_key="helper")
    assert "should_retrieve" in info
    assert "mode" in info


def test_controller_retrieves_when_uncertain():
    ctrl = MetaRAGController(confidence_threshold=0.58, enabled=True)
    d = ctrl.decide(
        "What is the population of Mumbai in 2024?",
        confidence=0.3,
        arousal=1.5,
        conflict=0.8,
        session_key="t2",
    )
    assert d.should_retrieve is True
    assert d.top_k >= 2


def test_controller_force_overrides():
    ctrl = MetaRAGController(enabled=True)
    on = ctrl.decide("x", force=True, session_key="t3")
    off = ctrl.decide("x", force=False, session_key="t3")
    assert on.should_retrieve is True
    assert off.should_retrieve is False


def test_controller_disabled():
    ctrl = MetaRAGController(enabled=False)
    d = ctrl.decide("What is AERIS?", session_key="t4")
    assert d.should_retrieve is False
    assert d.reason == "meta_rag_disabled"


def test_dynamic_top_k_increases_with_uncertainty():
    ctrl = MetaRAGController(min_top_k=2, max_top_k=8, confidence_threshold=0.9)
    low = ctrl.decide("q", confidence=0.8, arousal=0.1, conflict=0.1, session_key="a")
    high = ctrl.decide("q", confidence=0.1, arousal=2.0, conflict=1.5, session_key="b")
    assert high.top_k >= low.top_k


def test_get_meta_rag_singleton():
    a = get_meta_rag()
    b = get_meta_rag()
    assert a is b


def test_maybe_augment_no_hits_returns_original():
    """Empty store → prompt unchanged even if decision says retrieve."""
    ctrl = MetaRAGController(enabled=True)
    # Use a fresh controller with empty store by not loading default knowledge
    from neurofield.rag.pipeline import RAGPipeline

    ctrl._rag = RAGPipeline()  # empty
    prompt = "What is the capital of France?"
    out, decision, hits = ctrl.maybe_augment(
        prompt, confidence=0.1, arousal=1.5, conflict=1.0, session_key="empty"
    )
    assert hits == []
    assert out == prompt  # no hits above min_score → original prompt
    assert decision.should_retrieve is True
