"""Megha — defensive cybersecurity & authorized testing assistant for SENTINEL.

Scope (legal / ethical only):
  - Security review checklists and hardening guidance
  - Authorized assessment planning (requires explicit scope attestation)
  - Secure-code / config review heuristics
  - Incident triage & debug playbooks
  - High-level compliance process maps (not legal advice)

Out of scope:
  - Unauthorized access, exploit development, credential theft, malware
  - Attacking systems without documented permission
"""

from sentinel.megha.engine import (
    MeghaRequest,
    MeghaResult,
    run_megha,
    list_capabilities,
)
from sentinel.megha.authz import require_authorization, AuthorizationError

__all__ = [
    "MeghaRequest",
    "MeghaResult",
    "run_megha",
    "list_capabilities",
    "require_authorization",
    "AuthorizationError",
]
