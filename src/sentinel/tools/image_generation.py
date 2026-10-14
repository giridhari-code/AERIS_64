"""Image generation capability gate.

Providers are never hard-coded. Enable with NEUROFIELD_IMAGE_GEN=1 and set
NEUROFIELD_IMAGE_PROVIDER plus the provider credential env var. Until then
the tool reports unavailable and is omitted from the active catalog.
"""

from __future__ import annotations

import os
from typing import Optional

from sentinel.tools.config import load_tools_config


def image_generation_available() -> bool:
    cfg = load_tools_config()
    if not cfg.image_generation_enabled:
        return False
    provider = cfg.image_provider
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "fal":
        return bool(os.environ.get("FAL_KEY"))
    if provider == "replicate":
        return bool(os.environ.get("REPLICATE_API_TOKEN"))
    return False


def image_generation(
    prompt: str,
    size: str = "1024x1024",
    provider: Optional[str] = None,
) -> str:
    """
    Generate an image from ``prompt`` using the configured provider.

    Returns a status string. On success a production deployment should return
    a durable URL or storage key; this module only performs the capability
    check until a provider client is wired in deployment config.
    """
    cfg = load_tools_config()
    text = (prompt or "").strip()
    if not text:
        return "ImageGeneration: empty prompt."
    if len(text) > 2_000:
        return "ImageGeneration: prompt exceeds 2000 characters."

    if not cfg.image_generation_enabled:
        return (
            "ImageGeneration: disabled. "
            "Set NEUROFIELD_IMAGE_GEN=1 and NEUROFIELD_IMAGE_PROVIDER to enable."
        )

    chosen = (provider or cfg.image_provider or "").strip().lower()
    if not chosen:
        return (
            "ImageGeneration: NEUROFIELD_IMAGE_PROVIDER is not set "
            "(expected: openai | fal | replicate)."
        )

    if not image_generation_available():
        return (
            f"ImageGeneration: provider '{chosen}' is selected but credentials "
            "are missing in the environment."
        )

    # Provider HTTP clients belong in deployment-specific modules, not in the
    # core research tree. Fail closed with an explicit contract.
    return (
        f"ImageGeneration: provider '{chosen}' is configured but no client "
        "implementation is loaded in this process. "
        "Register a client adapter in the serving layer."
    )
