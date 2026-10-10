"""SDK client unit tests (httpx mock transport — no live server)."""

from __future__ import annotations

import json

import httpx
import pytest

from neurofield.sdk import AerisClient, AerisClientError


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/healthz":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/chat":
        body = json.loads(request.content.decode())
        assert "prompt" in body
        assert "max_new_tokens" in body
        return httpx.Response(
            200,
            json={
                "text": "I am AERIS_64.",
                "session_id": "abc12345deadbeef",
                "latency_ms": 12.5,
                "model": "test",
            },
        )
    if request.url.path == "/v1/tools/python":
        return httpx.Response(200, json={"ok": True, "stdout": "1\n"})
    if request.url.path == "/v1/reset":
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(404, json={"detail": "not found"})


def test_health_and_chat():
    transport = httpx.MockTransport(_handler)
    with AerisClient("http://test", transport=transport) as c:
        h = c.health()
        assert h.get("status") == "ok"
        r = c.chat("Who are you?", max_new_tokens=32, temperature=0.2)
        assert "AERIS" in r.text
        assert r.session_id
        assert r.latency_ms == 12.5


def test_error_surface():
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"msg": "bad"}]})

    transport = httpx.MockTransport(boom)
    with AerisClient("http://test", transport=transport) as c:
        with pytest.raises(AerisClientError) as ei:
            c.chat("x")
        assert ei.value.status_code == 422


def test_run_python():
    transport = httpx.MockTransport(_handler)
    with AerisClient("http://test", transport=transport) as c:
        out = c.run_python("print(1)")
        assert out.get("ok") is True
