"""Production runtime profile for SENTINEL API.

NEUROFIELD_ENV=production (or PRODUCTION=1) enables strict mode:
  - No demo checkpoint auto-fallback
  - API keys required
  - Metrics not public by default
  - Memory persistence requires keys
  - OpenAPI docs can be disabled
  - Stricter host binding guidance in logs

This does not magically improve model quality — that requires data + training.
It makes the *service* safe to expose behind a reverse proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _truthy(name: str, default: str = "0") -> bool:
    return (os.environ.get(name) or default).strip().lower() in {
        "1", "true", "yes", "on", "production", "prod",
    }


def is_production() -> bool:
    env = (os.environ.get("NEUROFIELD_ENV") or os.environ.get("ENV") or "").strip().lower()
    if env in {"production", "prod"}:
        return True
    return _truthy("PRODUCTION", "0") or _truthy("NEUROFIELD_PRODUCTION", "0")


@dataclass(frozen=True)
class ProductionProfile:
    production: bool
    require_api_keys: bool
    allow_demo_checkpoint: bool
    metrics_public: bool
    allow_open_memory: bool
    allow_open_tools: bool
    expose_docs: bool
    strict_load: bool

    def issues(self, has_api_keys: bool, checkpoint_set: bool) -> list[str]:
        problems: list[str] = []
        if not self.production:
            return problems
        if self.require_api_keys and not has_api_keys:
            problems.append("NEUROFIELD_API_KEYS required in production")
        if not checkpoint_set:
            problems.append("NEUROFIELD_CHECKPOINT required in production (no demo fallback)")
        if self.allow_open_memory:
            problems.append("NEUROFIELD_MEMORY_ALLOW_OPEN must be 0 in production")
        if self.allow_open_tools:
            problems.append("NEUROFIELD_TOOLS_ALLOW_OPEN must be 0 in production")
        return problems


def load_profile() -> ProductionProfile:
    prod = is_production()
    if prod:
        return ProductionProfile(
            production=True,
            require_api_keys=True,
            allow_demo_checkpoint=False,
            metrics_public=_truthy("NEUROFIELD_METRICS_PUBLIC", "0"),
            allow_open_memory=_truthy("NEUROFIELD_MEMORY_ALLOW_OPEN", "0"),
            allow_open_tools=_truthy("NEUROFIELD_TOOLS_ALLOW_OPEN", "0"),
            expose_docs=_truthy("NEUROFIELD_EXPOSE_DOCS", "0"),
            strict_load=_truthy("NEUROFIELD_STRICT_LOAD", "1"),
        )
    return ProductionProfile(
        production=False,
        require_api_keys=False,
        allow_demo_checkpoint=True,
        metrics_public=_truthy("NEUROFIELD_METRICS_PUBLIC", "1"),
        allow_open_memory=_truthy("NEUROFIELD_MEMORY_ALLOW_OPEN", "0"),
        allow_open_tools=_truthy("NEUROFIELD_TOOLS_ALLOW_OPEN", "0"),
        expose_docs=True,
        strict_load=_truthy("NEUROFIELD_STRICT_LOAD", "1"),
    )


def security_headers() -> dict[str, str]:
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "X-SENTINEL-Env": "production" if is_production() else "development",
    }
