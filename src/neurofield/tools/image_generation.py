"""Image Generation tool (interface + provider stub).

Real image models (Flux, SD3, DALL-E, Grok Imagine, etc.) need an API key.
This module provides a clean interface so the agent can call it; wire a provider
when credentials are available.
"""

from __future__ import annotations

import os
from typing import Optional


def image_generation(
    prompt: str,
    size: str = "1024x1024",
    provider: Optional[str] = None,
) -> str:
    """
    Generate an image from a text prompt.

    Returns a status message. When a real provider is configured via env
    (e.g. OPENAI_API_KEY, FAL_KEY, REPLICATE_API_TOKEN), this can be extended
    to call the provider and return a URL / local path.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return "ImageGeneration: empty prompt."
    if len(prompt) > 2_000:
        return "ImageGeneration: prompt too long."

    provider = (provider or os.environ.get("NEUROFIELD_IMAGE_PROVIDER") or "none").lower()

    # Detect common keys so the message is helpful
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_fal = bool(os.environ.get("FAL_KEY"))
    has_replicate = bool(os.environ.get("REPLICATE_API_TOKEN"))

    if provider == "none" and not (has_openai or has_fal or has_replicate):
        return (
            "ImageGeneration: no image provider configured.\n"
            "Set one of OPENAI_API_KEY / FAL_KEY / REPLICATE_API_TOKEN "
            "or NEUROFIELD_IMAGE_PROVIDER, then re-call this tool.\n"
            f"Prompt received: {prompt[:200]}"
        )

    # Placeholder — real HTTP call would go here
    return (
        f"ImageGeneration: provider='{provider}' not fully wired in this build.\n"
        f"Prompt: {prompt[:300]}\n"
        f"Size: {size}\n"
        "Add a concrete client (OpenAI images / fal.ai / Replicate) in this module to enable."
    )
