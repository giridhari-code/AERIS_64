"""Lightweight web search (DuckDuckGo HTML/lite, no key)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> list[dict[str, str]]:
    """
    Returns list of {title, snippet, url}.
    Uses DuckDuckGo lite HTML — best-effort, may change.
    """
    q = urllib.parse.urlencode({"q": query})
    url = f"https://lite.duckduckgo.com/lite/?{q}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AERIS-NeuroField/1.0; +research)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    results: list[dict[str, str]] = []
    # lite results: links with class result-link
    for m in re.finditer(
        r'href="(https?://[^"]+)"[^>]*>([^<]{5,120})</a>',
        html,
        flags=re.I,
    ):
        href, title = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        if "duckduckgo.com" in href:
            continue
        results.append({"title": title, "url": href, "snippet": title})
        if len(results) >= max_results:
            break

    if not results:
        # Instant Answer API fallback
        ia = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
        ia_url = f"https://api.duckduckgo.com/?{ia}"
        req2 = urllib.request.Request(ia_url, headers={"User-Agent": "AERIS-NeuroField/1.0"})
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        abstract = (data.get("AbstractText") or "").strip()
        heading = (data.get("Heading") or query).strip()
        abs_url = (data.get("AbstractURL") or "").strip()
        if abstract:
            results.append({"title": heading, "snippet": abstract[:400], "url": abs_url})
        for t in (data.get("RelatedTopics") or [])[: max_results - len(results)]:
            if isinstance(t, dict) and t.get("Text"):
                results.append(
                    {
                        "title": t.get("Text", "")[:80],
                        "snippet": t.get("Text", "")[:300],
                        "url": t.get("FirstURL") or "",
                    }
                )
    return results


def web_search_report(query: str, max_results: int = 5) -> str:
    try:
        hits = web_search(query, max_results=max_results)
        if not hits:
            return f"Web search: no results for '{query}'."
        lines = [f"Web search results for: {query}"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h.get('title', '')} — {h.get('snippet', '')[:200]} ({h.get('url', '')})")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search error: {e}"
