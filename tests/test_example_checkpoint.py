"""Smoke test: the shipped demo checkpoint (checkpoints/examples/AERIS_main) loads and serves."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sentinel.serving.server import create_app  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "examples" / "AERIS_main"


@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="demo weights not present")
def test_demo_checkpoint_loads_and_generates(monkeypatch):
    monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    c = TestClient(create_app(checkpoint=str(CKPT), device="cpu"))
    assert c.get("/readyz").status_code == 200
    r = c.post("/v1/completions", json={"input_ids": [1, 2, 3], "max_new_tokens": 8, "temperature": 0.0})
    assert r.status_code == 200, r.text
    assert len(r.json()["output_ids"]) >= 8


def test_checkpoint_without_slow_scale_loads_with_the_legacy_value_and_new_ones_keep_theirs(tmp_path):
    """Old checkpoints had no `slow_scale` (slow read unscaled = 1.0); a checkpoint that has it keeps its own value."""
    import json
    import shutil

    from safetensors.torch import load_file, save_file

    import torch

    assert CKPT.exists()
    legacy = tmp_path / "legacy"
    shutil.copytree(CKPT, legacy)
    weights = load_file(str(legacy / "model.safetensors"))
    weights.pop("slow_scale", None)                       # what a pre-slow_scale checkpoint looks like
    save_file(weights, str(legacy / "model.safetensors"))
    from sentinel.serving.server import create_app

    app = create_app(checkpoint=str(legacy), device="cpu")
    import sentinel.serving.server as srv

    assert float(srv._model.slow_scale) == 1.0

    modern = tmp_path / "modern"
    shutil.copytree(CKPT, modern)
    w2 = load_file(str(modern / "model.safetensors"))
    w2["slow_scale"] = torch.tensor(2.25)
    save_file(w2, str(modern / "model.safetensors"))
    create_app(checkpoint=str(modern), device="cpu")
    assert abs(float(srv._model.slow_scale) - 2.25) < 1e-6
    assert app is not None
