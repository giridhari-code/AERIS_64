"""Smoke test: the shipped demo checkpoint (checkpoints/examples/AERIS_main) loads and serves."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from neurofield.serving.server import create_app  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "examples" / "AERIS_main"


@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="demo weights not present")
def test_demo_checkpoint_loads_and_generates(monkeypatch):
    monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    c = TestClient(create_app(checkpoint=str(CKPT), device="cpu"))
    assert c.get("/readyz").status_code == 200
    r = c.post("/v1/completions", json={"input_ids": [1, 2, 3], "max_new_tokens": 8, "temperature": 0.0})
    assert r.status_code == 200, r.text
    assert len(r.json()["output_ids"]) >= 8
