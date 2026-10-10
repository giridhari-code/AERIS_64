"""FX rates via Frankfurter (ECB, no key)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request


def convert_currency(amount: float, fr: str, to: str, timeout: float = 8.0) -> str:
    fr, to = fr.upper().strip(), to.upper().strip()
    q = urllib.parse.urlencode({"amount": amount, "from": fr, "to": to})
    url = f"https://api.frankfurter.app/latest?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-NeuroField/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rates = data.get("rates") or {}
        if to not in rates:
            return f"Currency: no rate {fr}->{to}."
        return (
            f"Currency: {amount} {fr} = {rates[to]} {to} "
            f"(date {data.get('date')}, source Frankfurter/ECB)."
        )
    except Exception as e:
        return f"Currency error: {e}"
