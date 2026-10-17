"""KALO BOT identity — face of AERIS_64; core engine is SENTINEL; model AERIS_64."""

from __future__ import annotations

from typing import Any

PRODUCT_NAME = "KALO BOT"
TAGLINE = "Think Smarter. Do More."
BRAND = {
    "name": PRODUCT_NAME,
    "tagline": TAGLINE,
    "repo": "AERIS_64",
    "palette": {"void": "#0B0F1A", "kalo": "#6C5CE7", "signal": "#00E5C0"},
}
ENGINE = "SENTINEL"
MODEL_NAME = "AERIS_64"

CHILDREN = [
    {"id": "bhutu", "name": "Bhutu", "role": "Metacognition / traffic control"},
    {"id": "megha", "name": "Megha", "role": "Defensive security assistant"},
    {"id": "buddy", "name": "Buddy", "role": "Companion float (outside chat)"},
    {"id": "ajeebwork", "name": "AjeebWork", "role": "Plan · execute · file work"},
    {"id": "kalofacts", "name": "Kalofacts", "role": "Artifacts workspace"},
    {"id": "dataset_builder", "name": "Dataset Builder", "role": "Training data pipeline"},
]


def system_preamble() -> str:
    return (
        f"You are {PRODUCT_NAME}, the product face of AERIS_64. "
        f"Tagline: {TAGLINE}. "
        f"Core engine: {ENGINE}. Model name: {MODEL_NAME}. "
        "Be direct, helpful, and honest about limits."
    )


def identity_payload() -> dict[str, Any]:
    return {
        "name": PRODUCT_NAME,
        "tagline": TAGLINE,
        "brand": BRAND,
        "engine": ENGINE,
        "model": MODEL_NAME,
        "repo": "AERIS_64",
        "preamble": system_preamble(),
        "children": CHILDREN,
        "endpoints": {
            "chat": "/v1/chat",
            "identity": "/v1/kalo",
            "dataset_builder": "/v1/tools/dataset_builder",
            "buddy": "/v1/buddy",
        },
    }
