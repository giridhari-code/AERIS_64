"""Frontier-style API entry (re-exports serving)."""

from sentinel.serving.server import create_app

__all__ = ["create_app"]
