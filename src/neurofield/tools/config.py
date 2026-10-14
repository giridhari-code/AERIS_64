"""Production configuration for NeuroField external tools.

All limits and feature flags come from environment variables so the same
binary can run in dev, staging, and production without code changes.
No provider keys or paths are hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ToolsConfig:
    """Immutable tool runtime settings (read once per process)."""

    # Master switch (also used by orchestrate.tools_enabled)
    enabled: bool = True

    # Sandbox-backed tools
    code_timeout_sec: float = 12.0
    code_max_output_bytes: int = 48_000
    code_max_memory_mb: int = 384
    code_max_source_chars: int = 12_000

    bash_timeout_sec: float = 10.0
    bash_max_output_bytes: int = 40_000
    bash_max_memory_mb: int = 256
    bash_max_command_chars: int = 2_000

    # Network tools
    web_fetch_timeout_sec: float = 12.0
    web_fetch_max_chars: int = 8_000
    web_fetch_max_download_bytes: int = 1_500_000

    # File search
    file_search_max_results: int = 8
    file_search_max_file_bytes: int = 400_000
    file_search_max_snippet: int = 180

    # Optional capabilities (off until explicitly enabled + credentials present)
    image_generation_enabled: bool = False
    computer_use_enabled: bool = False
    image_provider: str = ""  # openai | fal | replicate | ""

    @classmethod
    def from_env(cls) -> "ToolsConfig":
        return cls(
            enabled=_env_bool("NEUROFIELD_WEB_TOOLS", True),
            code_timeout_sec=_env_float("NEUROFIELD_CODE_TIMEOUT", 12.0),
            code_max_output_bytes=_env_int("NEUROFIELD_CODE_MAX_OUTPUT", 48_000),
            code_max_memory_mb=_env_int("NEUROFIELD_CODE_MAX_MEMORY_MB", 384),
            code_max_source_chars=_env_int("NEUROFIELD_CODE_MAX_SOURCE", 12_000),
            bash_timeout_sec=_env_float("NEUROFIELD_BASH_TIMEOUT", 10.0),
            bash_max_output_bytes=_env_int("NEUROFIELD_BASH_MAX_OUTPUT", 40_000),
            bash_max_memory_mb=_env_int("NEUROFIELD_BASH_MAX_MEMORY_MB", 256),
            bash_max_command_chars=_env_int("NEUROFIELD_BASH_MAX_COMMAND", 2_000),
            web_fetch_timeout_sec=_env_float("NEUROFIELD_FETCH_TIMEOUT", 12.0),
            web_fetch_max_chars=_env_int("NEUROFIELD_FETCH_MAX_CHARS", 8_000),
            web_fetch_max_download_bytes=_env_int(
                "NEUROFIELD_FETCH_MAX_DOWNLOAD", 1_500_000
            ),
            file_search_max_results=_env_int("NEUROFIELD_FILE_SEARCH_MAX", 8),
            file_search_max_file_bytes=_env_int(
                "NEUROFIELD_FILE_SEARCH_MAX_FILE", 400_000
            ),
            file_search_max_snippet=_env_int("NEUROFIELD_FILE_SEARCH_SNIPPET", 180),
            image_generation_enabled=_env_bool("NEUROFIELD_IMAGE_GEN", False),
            computer_use_enabled=_env_bool("NEUROFIELD_COMPUTER_USE", False),
            image_provider=(
                os.environ.get("NEUROFIELD_IMAGE_PROVIDER") or ""
            ).strip().lower(),
        )


# Process-wide default; tests can replace via load_tools_config()
_CONFIG: Optional[ToolsConfig] = None


def load_tools_config(force: bool = False) -> ToolsConfig:
    global _CONFIG
    if _CONFIG is None or force:
        _CONFIG = ToolsConfig.from_env()
    return _CONFIG
