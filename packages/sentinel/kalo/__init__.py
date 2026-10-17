"""KALO BOT — mother agent (product face of AERIS_64)."""

from sentinel.kalo.identity import (
    PRODUCT_NAME,
    TAGLINE,
    BRAND,
    ENGINE,
    MODEL_NAME,
    system_preamble,
    identity_payload,
)
from sentinel.kalo.router import Route, route_intent
from sentinel.kalo.agent import KaloBot, KaloTurn, run_kalo

__all__ = [
    "PRODUCT_NAME",
    "TAGLINE",
    "BRAND",
    "ENGINE",
    "MODEL_NAME",
    "system_preamble",
    "identity_payload",
    "Route",
    "route_intent",
    "KaloBot",
    "KaloTurn",
    "run_kalo",
]
