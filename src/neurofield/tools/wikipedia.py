"""Wikipedia summary (no key)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request


def wikipedia_summary(topic: str, lang: str = "en", timeout: float = 10.0) -> str:
    topic = (topic or "").strip()
    if not topic:
        return "Wikipedia: empty topic."
    title = urllib.parse.quote(topic.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-NeuroField/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        extract = (data.get("extract") or "").strip()
        page = data.get("content_urls", {}).get("desktop", {}).get("page") or url
        if not extract:
            return f"Wikipedia: no summary for '{topic}'."
        return f"Wikipedia ({lang}): {data.get('title', topic)}\n{extract[:800]}\nSource: {page}"
    except Exception as e:
        return f"Wikipedia error: {e}"
