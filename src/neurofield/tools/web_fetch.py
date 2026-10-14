"""Web Fetch — download a URL and return readable text (no JS rendering)."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional


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
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def web_fetch(url: str, max_chars: int = 8_000, timeout: float = 12.0) -> str:
    """
    Fetch a public URL and return a plain-text summary of the page.
    """
    url = (url or "").strip()
    if not url:
        return "WebFetch: empty URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "WebFetch: only http/https allowed."

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AERIS-NeuroField/1.0; +research)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            # Limit download size
            if len(raw) > 1_500_000:
                raw = raw[:1_500_000]
            charset = "utf-8"
            if "charset=" in content_type:
                m = re.search(r"charset=([^\s;]+)", content_type, re.I)
                if m:
                    charset = m.group(1).strip("'\"")
            html = raw.decode(charset, errors="ignore")
    except Exception as e:
        return f"WebFetch error: {e}"

    if "html" in content_type.lower() or "<html" in html[:500].lower():
        text = _html_to_text(html)
    else:
        text = html

    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n... [truncated]"

    return f"WebFetch ({url}):\n{text or '(empty page)'}"
