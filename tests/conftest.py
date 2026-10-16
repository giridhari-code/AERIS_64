"""Shared pytest setup.

The FastAPI app uses one process-wide ``RateLimiter`` (NEUROFIELD_RATE_LIMIT,
default 60 requests / 60 s). The unit-test suite fires far more requests than
that in a few seconds — from several ``create_app`` instances sharing the
same global limiter — which used to 429 mid-suite (KeyError-style failures in
later files). Raise the ceiling for the test session; the limiter's own
behavior is covered directly in ``tests/test_company.py`` (its own instances).
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _relax_rate_limit():
    try:
        from sentinel.serving import server as server_mod

        server_mod._limiter.max_requests = 10**9
    except Exception:  # noqa: BLE001, S110 - optional deps missing → those tests skip anyway
        pass
    yield
