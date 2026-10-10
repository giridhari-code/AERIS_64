"""Production HTTP client for a running NeuroField / AERIS server.

Configuration (no hardcoded host in call sites):

  base_url  — constructor arg, or env NEUROFIELD_BASE_URL (default http://127.0.0.1:8000)
  api_key   — constructor arg, or env NEUROFIELD_API_KEY (optional; sent as X-API-Key)

Requires: pip install httpx  (neurofield[sdk] or neurofield[serve])
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

try:
    import httpx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "neurofield.sdk requires httpx. Install with: pip install 'neurofield[sdk]'"
    ) from e


class AerisClientError(RuntimeError):
    """Raised when the server returns a non-success response or transport fails."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ChatResult:
    text: str
    session_id: str
    latency_ms: float
    model: str
    raw: Mapping[str, Any]


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if isinstance(v, str) and v.strip() else default


class AerisClient:
    """Thread-friendly thin client over the public HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        url = (base_url or _env("NEUROFIELD_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        key = api_key if api_key is not None else _env("NEUROFIELD_API_KEY", "")
        headers: dict[str, str] = {"Accept": "application/json"}
        if key:
            headers["X-API-Key"] = key
        self._base = url
        self._client = httpx.Client(
            base_url=url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AerisClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- low level ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            r = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise AerisClientError(f"transport error: {e}") from e
        if r.status_code >= 400:
            body: Any
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise AerisClientError(
                f"{method} {path} → {r.status_code}",
                status_code=r.status_code,
                body=body,
            )
        if r.status_code == 204 or not r.content:
            return None
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            return r.json()
        return r.text

    # --- health / meta -----------------------------------------------------

    def health(self) -> dict[str, Any]:
        out = self._request("GET", "/healthz")
        return out if isinstance(out, dict) else {"raw": out}

    def ready(self) -> dict[str, Any]:
        out = self._request("GET", "/readyz")
        return out if isinstance(out, dict) else {"raw": out}

    def models(self) -> Any:
        return self._request("GET", "/v1/models")

    # --- chat / completion -------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        auto_length: bool = True,
        session_id: str | None = None,
        postprocess: bool = True,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "auto_length": auto_length,
            "postprocess": postprocess,
        }
        if session_id:
            payload["session_id"] = session_id
        data = self._request("POST", "/v1/chat", json=payload)
        if not isinstance(data, dict):
            raise AerisClientError("chat response is not a JSON object", body=data)
        return ChatResult(
            text=str(data.get("text", "")),
            session_id=str(data.get("session_id", "")),
            latency_ms=float(data.get("latency_ms") or 0.0),
            model=str(data.get("model") or ""),
            raw=data,
        )

    def complete(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int = 48,
        temperature: float = 1.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if session_id:
            payload["session_id"] = session_id
        data = self._request("POST", "/v1/completions", json=payload)
        if not isinstance(data, dict):
            raise AerisClientError("completions response is not a JSON object", body=data)
        return data

    def reset(self, session_id: str) -> dict[str, Any]:
        data = self._request("POST", "/v1/reset", params={"session_id": session_id})
        return data if isinstance(data, dict) else {"raw": data}

    # --- tools (server must have sandbox enabled) --------------------------

    def run_python(self, code: str, *, session_id: str = "default") -> dict[str, Any]:
        data = self._request(
            "POST", "/v1/tools/python", json={"code": code, "session_id": session_id}
        )
        return data if isinstance(data, dict) else {"raw": data}

    def run_shell(self, command: str, *, session_id: str = "default") -> dict[str, Any]:
        data = self._request(
            "POST", "/v1/tools/shell", json={"command": command, "session_id": session_id}
        )
        return data if isinstance(data, dict) else {"raw": data}

    def tools_audit(self, session_id: str = "default") -> dict[str, Any]:
        data = self._request("GET", "/v1/tools/audit", params={"session_id": session_id})
        return data if isinstance(data, dict) else {"raw": data}

    def tools_reset(self, session_id: str = "default") -> dict[str, Any]:
        data = self._request("POST", "/v1/tools/reset", json={"session_id": session_id})
        return data if isinstance(data, dict) else {"raw": data}
