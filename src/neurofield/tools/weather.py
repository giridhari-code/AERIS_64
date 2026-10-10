"""Weather via Open-Meteo (no API key)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Optional


def geocode(place: str, timeout: float = 8.0) -> Optional[dict[str, Any]]:
    q = urllib.parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-NeuroField/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return {
        "name": r.get("name"),
        "country": r.get("country"),
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
    }


def forecast(lat: float, lon: float, timeout: float = 8.0) -> dict[str, Any]:
    q = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-NeuroField/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


_WMO = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    61: "rain",
    63: "rain",
    65: "heavy rain",
    71: "snow",
    80: "rain showers",
    95: "thunderstorm",
}


def weather_report(place: str) -> str:
    """Human-readable weather string for tool context."""
    try:
        g = geocode(place)
        if not g:
            return f"Weather: could not find location '{place}'."
        data = forecast(float(g["latitude"]), float(g["longitude"]))
        cur = data.get("current") or {}
        code = int(cur.get("weather_code") or 0)
        desc = _WMO.get(code, f"code {code}")
        temp = cur.get("temperature_2m")
        hum = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        where = f"{g.get('name')}, {g.get('country')}"
        return (
            f"Weather now in {where}: {temp}°C, {desc}, "
            f"humidity {hum}%, wind {wind} km/h (source: Open-Meteo)."
        )
    except Exception as e:
        return f"Weather tool error: {e}"
