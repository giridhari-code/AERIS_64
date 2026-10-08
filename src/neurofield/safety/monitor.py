"""Safety monitoring and audit utilities."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Online monitor for the signals defined in Section 6 of the report.
    Detects anomalies via running z-score and can trigger a freeze.
    """

    def __init__(
        self,
        window: int = 500,
        z_threshold: float = 4.0,
        log_path: Optional[str | Path] = None,
    ):
        self.window = window
        self.z_threshold = z_threshold
        self.history: dict[str, Deque[float]] = {
            "surprise": deque(maxlen=window),
            "gate": deque(maxlen=window),
            "write_norm": deque(maxlen=window),
            "mem_norm": deque(maxlen=window),
            "router_ent": deque(maxlen=window),
        }
        self.log_path = Path(log_path) if log_path else None
        self.frozen = False
        self.anomaly_count = 0

    def update(self, audit: dict[str, Any]) -> list[str]:
        """Ingest one step of audit signals. Returns list of triggered alarms."""
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
            if vals:
                v = float(vals[-1]) if isinstance(vals, list) else float(vals)
                self.history[key].append(v)
                if len(self.history[key]) >= 30:
                    mean = sum(self.history[key]) / len(self.history[key])
                    var = sum((x - mean) ** 2 for x in self.history[key]) / len(
                        self.history[key]
                    )
                    std = max(var ** 0.5, 1e-6)
                    z = abs(v - mean) / std
                    if z > self.z_threshold:
                        alarms.append(f"{key}_z={z:.2f}")
                        self.anomaly_count += 1

        if self.log_path and audit:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit) + "\n")

        return alarms

    def should_freeze(self) -> bool:
        return self.anomaly_count > 5
