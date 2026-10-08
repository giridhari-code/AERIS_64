"""Smoke tests for platform pieces (tokenizers, normaliser, rate limiter, auth)."""

import pytest
from fastapi import HTTPException

from neurofield.serving.auth import RateLimiter, _key_valid
from neurofield.tokenizer.bpe import BPETokenizer
from neurofield.tokenizer.char import CharTokenizer
from neurofield.utils.text_norm import normalize_prompt


def test_char_roundtrip():
    t = CharTokenizer.from_text("hello namaste")
    assert t.decode(t.encode("hello")) == "hello"


def test_bpe_train_encode():
    text = "hello hello world world namaste namaste " * 20
    tok = BPETokenizer.train(text, vocab_size=80, min_freq=1)
    ids = tok.encode("hello world")
    assert isinstance(ids, list) and len(ids) >= 1
    assert isinstance(tok.decode(ids), str)


def test_normalize_hindlish():
    # the old assertion ended in `or normalize_prompt(...)` (a non-empty string) so it could never fail
    assert normalize_prompt("namste") == "namaste"
    assert normalize_prompt("plz help") == "please help"
    assert normalize_prompt("vanakam") == "vanakkam"
    assert normalize_prompt("hello") == "hello"  # untouched


def test_rate_limiter():
    lim = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        lim.check("u1")
    with pytest.raises(HTTPException) as e:
        lim.check("u1")
    assert e.value.status_code == 429
    lim.check("u2")  # other keys unaffected


def test_rate_limiter_gc_bounds_memory():
    import time

    lim = RateLimiter(max_requests=1, window_seconds=0.05, max_keys=10)
    for i in range(50):
        lim.check(f"k{i}")
    time.sleep(0.1)  # all earlier windows expired
    lim.check("fresh")  # len > max_keys triggers GC of expired keys
    assert len(lim._hits) <= 2


def test_key_validation_constant_time_api():
    keys = {"alpha-secret", "beta-secret"}
    assert _key_valid("alpha-secret", keys)
    assert not _key_valid("alpha-secre", keys)
    assert not _key_valid("", keys)
