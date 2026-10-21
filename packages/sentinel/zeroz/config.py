"""0z code runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from sentinel.zeroz.policy import ALLOWED_TOOLS, PRODUCT_NAME


@dataclass
class ZeroZConfig:
    product_name: str = PRODUCT_NAME
    max_steps: int = 6
    max_tool_failures: int = 3
    tool_timeout_s: float = 15.0
    allowed_tools: tuple[str, ...] = tuple(sorted(ALLOWED_TOOLS))
    require_tool_for_live_facts: bool = True
    always_enrich: bool = True
    use_auto_rag: bool = True
    # Optional LM polish (checkpoint generate_fn); None = facts-only / template answer
    use_model_generate: bool = True
    checkpoint: Optional[str] = None
    device: str = "cpu"
    audit_dir: Optional[str] = None
    workspace: Optional[str] = None  # default cwd for file_search

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ZeroZConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        kw = {k: v for k, v in (d or {}).items() if k in known}
        if "allowed_tools" in kw and isinstance(kw["allowed_tools"], list):
            kw["allowed_tools"] = tuple(kw["allowed_tools"])
        return cls(**kw)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ZeroZConfig":
        if path is None:
            for cand in ("configs/zeroz.yaml", "configs/zeroz/default.yaml"):
                p = Path(cand)
                if p.is_file():
                    path = p
                    break
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw.get("zeroz") or raw)


def load_zeroz_config(path: str | Path | None = None) -> ZeroZConfig:
    return ZeroZConfig.load(path)
