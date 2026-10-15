"""Uvicorn target: ``uvicorn sentinel.api.app:create_app --factory``."""

from sentinel.serving.server import create_app

__all__ = ["create_app"]
