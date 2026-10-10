"""API key auth + in-memory rate limiting.

Notes
-----
* Keys are accepted ONLY via the ``X-API-Key`` header (query-string keys leak
  into proxy / access logs and were removed).
* Comparison is constant-time.
* The rate limiter is per-process. With several workers/replicas put a shared
  limiter (nginx, envoy, redis) in front; this one is a safety net.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque
from typing import Optional

from fastapi import Header, HTTPException, Request


def get_api_keys() -> set[str]:
    raw = os.environ.get("NEUROFIELD_API_KEYS", "")
    # empty set => auth disabled (dev mode)
    return {k.strip() for k in raw.split(",") if k.strip()}


def _key_valid(candidate: str, keys: set[str]) -> bool:
    ok = False
    for k in keys:  # no early exit: constant-time per key
        ok |= hmac.compare_digest(candidate.encode(), k.encode())
    return ok


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[str]:
    keys = get_api_keys()
    if not keys:
        return None  # open mode (development)
    if not x_api_key or not _key_valid(x_api_key, keys):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


async def require_api_key_strict(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Like ``require_api_key`` but NEVER open: for code-execution (sandbox) endpoints.

    With no NEUROFIELD_API_KEYS configured the endpoints answer 503 instead of running
    arbitrary commands for anyone who can reach the port. For local development only,
    set NEUROFIELD_TOOLS_ALLOW_OPEN=1 to allow unauthenticated use (returns "").
    """
    keys = get_api_keys()
    if not keys:
        if os.environ.get("NEUROFIELD_TOOLS_ALLOW_OPEN", "0") == "1":
            return ""
        raise HTTPException(
            status_code=503,
            detail="Sandbox tools are disabled until NEUROFIELD_API_KEYS is set "
            "(dev only: NEUROFIELD_TOOLS_ALLOW_OPEN=1).",
        )
    if not x_api_key or not _key_valid(x_api_key, keys):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


def client_ip(request: Request) -> str:
    """Client IP. Honours X-Forwarded-For only if NEUROFIELD_TRUST_PROXY=1."""
    if os.environ.get("NEUROFIELD_TRUST_PROXY", "0") == "1":
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "anon"


class RateLimiter:
    """Sliding-window per-key (or IP) rate limiter, thread-safe, bounded."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0, max_keys: int = 10_000):
        self.max_requests = max_requests
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self._lock:
            if len(self._hits) > self.max_keys:
                self._gc(now)
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            q.append(now)

    def _gc(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] > self.window]:
            self._hits.pop(k, None)


def rate_limit_from_env() -> RateLimiter:
    max_r = int(os.environ.get("NEUROFIELD_RATE_LIMIT", "120"))
    win = float(os.environ.get("NEUROFIELD_RATE_WINDOW", "60"))
    return RateLimiter(max_requests=max_r, window_seconds=win)
