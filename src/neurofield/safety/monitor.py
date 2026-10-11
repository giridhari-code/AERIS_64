"""Safety observability (AESC paper Section 7).

Per-token / per-turn audit → z-score anomaly → optional memory freeze.
When frozen, the serve path must set gate g=0 so fast M only decays (no new writes).
Replay into shared slow weights only from audited training — never from live flagged sessions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional


class SafetyMonitor:
    def __init__(
        self,
        z_threshold: float = 4.0,
        log_path: Optional[str] = None,
        freeze_after_alarms: int = 5,
    ):
        self.z_threshold = z_threshold
        self.log_path = log_path
        self.freeze_after_alarms = max(1, int(freeze_after_alarms))
        self.history: dict[str, list[float]] = defaultdict(list)
        self.anomaly_count = 0
        self.last_alarms: list[str] = []
        self.frozen = False

    def update(self, audit: Optional[dict[str, Any]]) -> list[str]:
        alarms: list[str] = []
        if not audit:
            return alarms

        mapping = {
            "gate": "gates",
            "write_norm": "write_norms",
            "mem_norm": "mem_norms",
            "router_ent": "router_entropy",
        }
        for key, audit_key in mapping.items():
            vals = audit.get(audit_key)
            if vals is None:
                continue
            if isinstance(vals, list) and vals:
                v = float(vals[-1])
            else:
                try:
                    v = float(vals)
                except (TypeError, ValueError):
                    continue
            self.history[key].append(v)
            if len(self.history[key]) >= 30:
                mean = sum(self.history[key]) / len(self.history[key])
                var = sum((x - mean) ** 2 for x in self.history[key]) / len(self.history[key])
                std = max(var ** 0.5, 1e-6)
                z = abs(v - mean) / std
                if z > self.z_threshold:
                    alarms.append(f"{key}_z={z:.2f}")
                    self.anomaly_count += 1

        self.last_alarms = alarms
        if self.anomaly_count >= self.freeze_after_alarms:
            self.frozen = True

        if self.log_path and audit:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**audit, "alarms": alarms, "frozen": self.frozen}) + "\n")
            except OSError:
                pass

        return alarms

    def should_freeze(self) -> bool:
        """Paper: raised flag → freeze memory writes (g=0)."""
        return self.frozen or self.anomaly_count >= self.freeze_after_alarms

    def force_freeze(self) -> None:
        self.frozen = True
        self.anomaly_count = max(self.anomaly_count, self.freeze_after_alarms)

    def clear_freeze(self) -> None:
        self.frozen = False
        self.anomaly_count = 0
        self.last_alarms = []

    def reset_session_baselines(self) -> None:
        """New user session: keep global history optional; clear freeze flag."""
        self.clear_freeze()
