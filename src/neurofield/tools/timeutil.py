"""Current date/time (server clock + optional timezone)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def now_report(tz_name: Optional[str] = None) -> str:
    try:
        if tz_name:
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            label = tz_name
        else:
            now = datetime.now(timezone.utc).astimezone()
            label = str(now.tzinfo) if now.tzinfo else "local"
        return (
            f"Current datetime ({label}): {now.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"(weekday {now.strftime('%A')})."
        )
    except Exception as e:
        return f"Time tool error: {e}"
