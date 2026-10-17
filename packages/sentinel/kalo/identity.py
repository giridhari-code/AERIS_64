"""KALO BOT identity constants — product face of AERIS_64."""

from __future__ import annotations

from typing import Any

PRODUCT_NAME = "KALO BOT"
TAGLINE = "Think Smarter. Do More."
ENGINE = "SENTINEL"
MODEL_NAME = "AERIS_64"
REPO = "AERIS_64"

BRAND = {
    "name": PRODUCT_NAME,
    "tagline": TAGLINE,
    "repo": REPO,
    "palette": {"void": "#0B0F1A", "kalo": "#6C5CE7", "signal": "#00E5C0"},
}

# Capability registry (ids must match router targets / package names)
CHILDREN: list[dict[str, str]] = [
    {"id": "bhutu", "name": "Bhutu", "role": "Metacognition and tool policy"},
    {"id": "megha", "name": "Megha", "role": "Defensive security assistant"},
    {"id": "buddy", "name": "Buddy", "role": "Companion (float UI, outside chat)"},
    {"id": "ajeebwork", "name": "AjeebWork", "role": "Plan and file workstreams"},
    {"id": "kalofacts", "name": "Kalofacts", "role": "Artifacts workspace"},
    {"id": "dataset_builder", "name": "Dataset Builder", "role": "Training data pipeline"},
]


def system_preamble() -> str:
    return (
        f"You are {PRODUCT_NAME}, the product face of {REPO}. "
        f"Tagline: {TAGLINE}. Core engine: {ENGINE}. Model: {MODEL_NAME}. "
        "Be direct and factual. Prefer tools for live facts. Do not invent citations."
    )


def identity_payload() -> dict[str, Any]:
    return {
        "name": PRODUCT_NAME,
        "tagline": TAGLINE,
        "brand": BRAND,
        "engine": ENGINE,
        "model": MODEL_NAME,
        "repo": REPO,
        "preamble": system_preamble(),
        "children": list(CHILDREN),
        "endpoints": {
            "identity": "GET /v1/kalo",
            "run": "POST /v1/kalo/run",
            "chat": "POST /v1/chat",
            "dataset_builder": "POST /v1/tools/dataset_builder",
            "buddy": "/v1/buddy",
        },
    }
