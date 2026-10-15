"""Authorization gate for Megha assessment modes."""

from __future__ import annotations


class AuthorizationError(PermissionError):
    """Raised when assessment mode is requested without valid attestation."""


def require_authorization(
    *,
    authorized: bool,
    scope: str,
    attestor: str,
    purpose: str = "authorized security testing",
) -> dict:
    """
    Assessment / scan-planning modes require explicit attestation.

    Returns a signed-style record (plain dict) for audit logs.
    Does not perform any network activity itself.
    """
    scope = (scope or "").strip()
    attestor = (attestor or "").strip()
    if not authorized:
        raise AuthorizationError(
            "Megha assessment modes require authorized=true and a documented scope. "
            "Only use on systems you own or have written permission to test."
        )
    if len(scope) < 8:
        raise AuthorizationError("scope must describe the target system and limits (≥ 8 chars).")
    if len(attestor) < 2:
        raise AuthorizationError("attestor (name/team responsible) is required.")
    return {
        "authorized": True,
        "scope": scope,
        "attestor": attestor,
        "purpose": purpose,
        "policy": "defensive_and_authorized_only",
    }
