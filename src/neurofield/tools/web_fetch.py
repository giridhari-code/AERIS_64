"""Fetch a public HTTP(S) URL and return bounded plain text.

No JavaScript rendering. Intended for documentation pages, raw text, and
simple HTML. Download size and returned text length are config-capped.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from neurofield.tools.config import load_tools_config


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = False
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t + " ")


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def web_fetch(
    url: str,
    max_chars: int | None = None,
    timeout: float | None = None,
) -> str:
    """
    Download ``url`` and return a plain-text extract.

    Only ``http`` and ``https`` are accepted. Response body is truncated
    according to ``ToolsConfig``.
    """
    cfg = load_tools_config()
    max_chars = int(max_chars if max_chars is not None else cfg.web_fetch_max_chars)
    timeout = float(timeout if timeout is not None else cfg.web_fetch_timeout_sec)

    raw_url = (url or "").strip()
    if not raw_url:
        return "WebFetch: empty URL."
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url

    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        return "WebFetch: only http/https schemes are allowed."
    if not parsed.netloc:
        return "WebFetch: invalid URL (missing host)."

    try:
        req = urllib.request.Request(
            raw_url,
            headers={
                "User-Agent": "AERIS-NeuroField/1.0 (+research; web_fetch)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read(cfg.web_fetch_max_download_bytes + 1)
            if len(data) > cfg.web_fetch_max_download_bytes:
                data = data[: cfg.web_fetch_max_download_bytes]
            charset = "utf-8"
            m = re.search(r"charset=([^\s;]+)", content_type)
            if m:
                charset = m.group(1).strip("'\"").lower()
            body = data.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        return f"WebFetch HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"WebFetch network error: {e.reason}"
    except Exception as e:
        return f"WebFetch error: {type(e).__name__}: {e}"

    if "html" in content_type or body.lstrip()[:32].lower().startswith(
        ("<!doctype", "<html")
    ):
        text = _html_to_text(body)
    else:
        text = body

    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n... [truncated]"

    return f"WebFetch ({raw_url}):\n{text or '(empty body)'}"
