#!/usr/bin/env python3
"""Fail-closed checklist before exposing SENTINEL to a network."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from sentinel.serving.production import load_profile, is_production  # noqa: E402


def main() -> int:
    os.environ.setdefault("NEUROFIELD_ENV", os.environ.get("NEUROFIELD_ENV", "production"))
    profile = load_profile()
    keys = (os.environ.get("NEUROFIELD_API_KEYS") or os.environ.get("API_KEYS") or "").strip()
    ckpt = (os.environ.get("NEUROFIELD_CHECKPOINT") or os.environ.get("CHECKPOINT") or "").strip()
    issues = profile.issues(has_api_keys=bool(keys), checkpoint_set=bool(ckpt))

    if ckpt:
        p = Path(ckpt)
        if not (p / "model.safetensors").is_file() and not list(p.glob("*.safetensors")):
            issues.append(f"checkpoint folder missing weights: {ckpt}")

    print("env_production", is_production())
    print("profile", profile)
    if issues:
        print("FAIL")
        for i in issues:
            print(" -", i)
        return 1
    print("OK production gates satisfied")
    print("NOTE: model quality still depends on your trained weights and eval — not this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
