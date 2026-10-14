"""Safety observability (AESC paper Section 7).

Per-token / per-turn audit → z-score anomaly → optional memory freeze.
When frozen, the serve path sets state['memory_frozen']=True so the model
forces gate g=0 (fast M only decays; no new writes; slow ring does not snapshot).

Replay into shared slow weights only from audited training — never from live
flagged sessions (server refuses to persist slow ring when sess['anomaly']).

Baselines and freeze flags are **per session_id** so one user cannot freeze another.
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
        freeze_after_alarms: int = 3,
        min_history: int = 30,
        max_history: int = 500,
    ):
        self.z_threshold = float(z_threshold)
        self.log_path = log_path
        self.freeze_after_alarms = max(1, int(freeze_after_alarms))
        self.min_history = max(5, int(min_history))
        self.max_history = max(self.min_history, int(max_history))
        # session_id -> metric -> list[float]
        self.history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.anomaly_count: dict[str, int] = defaultdict(int)
        self.last_alarms: dict[str, list[str]] = defaultdict(list)
        self.frozen: dict[str, bool] = defaultdict(bool)

    def _sid(self, session_id: Optional[str]) -> str:
        return session_id or "_global"

    def update(
        self,
        audit: Optional[dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Ingest one turn's audit; return list of alarm strings for this update."""
        alarms: list[str] = []
        if not audit:
            return alarms
        sid = self._sid(session_id)
        hist = self.history[sid]

        mapping = {
            "gate": "gates",
            "write_norm": "write_norms",
            "mem_norm": "mem_norms",
            "router_ent": "router_entropy",
            "surprise": "surprises",
        }
        for key, audit_key in mapping.items():
            vals = audit.get(audit_key)
            if vals is None and key == "surprise":
                # fallback scalar from metrics-style audit
                vals = audit.get("mean_S")
            if vals is None:
                continue
            if isinstance(vals, list) and vals:
                # use mean of the turn (stable) + last for spike sensitivity
                try:
                    nums = [float(x) for x in vals]
                except (TypeError, ValueError):
                    continue
                v_mean = sum(nums) / len(nums)
                v_last = nums[-1]
                samples = (v_mean, v_last)
            else:
                try:
                    samples = (float(vals),)
                except (TypeError, ValueError):
                    continue

            for v in samples:
                hist[key].append(v)
                if len(hist[key]) > self.max_history:
                    hist[key] = hist[key][-self.max_history :]
                if len(hist[key]) >= self.min_history:
                    series = hist[key]
                    mean = sum(series) / len(series)
                    var = sum((x - mean) ** 2 for x in series) / len(series)
                    std = max(var ** 0.5, 1e-6)
                    z = abs(v - mean) / std
                    if z > self.z_threshold:
                        tag = f"{key}_z={z:.2f}"
                        if tag not in alarms:
                            alarms.append(tag)
                        self.anomaly_count[sid] += 1

        # hard caps from paper §7 (optional absolute trips)
        try:
            final_mem = float(audit.get("final_mem_norm") or 0.0)
            if final_mem > 1e6:  # pathological
                alarms.append(f"mem_norm_abs={final_mem:.2e}")
                self.anomaly_count[sid] += 1
        except (TypeError, ValueError):
            pass

        self.last_alarms[sid] = alarms
        if self.anomaly_count[sid] >= self.freeze_after_alarms:
            self.frozen[sid] = True

        if self.log_path and audit:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                **{k: audit.get(k) for k in ("gates", "write_norms", "mem_norms", "router_entropy", "surprises")},
                                "session_id": sid,
                                "alarms": alarms,
                                "frozen": self.frozen[sid],
                                "anomaly_count": self.anomaly_count[sid],
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass

        return alarms

    def should_freeze(self, session_id: Optional[str] = None) -> bool:
        """Paper: raised flag → freeze memory writes (g=0)."""
        sid = self._sid(session_id)
        return bool(self.frozen[sid] or self.anomaly_count[sid] >= self.freeze_after_alarms)

    def force_freeze(self, session_id: Optional[str] = None) -> None:
        sid = self._sid(session_id)
        self.frozen[sid] = True
        self.anomaly_count[sid] = max(self.anomaly_count[sid], self.freeze_after_alarms)

    def clear_freeze(self, session_id: Optional[str] = None) -> None:
        sid = self._sid(session_id)
        self.frozen[sid] = False
        self.anomaly_count[sid] = 0
        self.last_alarms[sid] = []

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """New chat session: clear freeze and baselines for this session only."""
        sid = self._sid(session_id)
        self.clear_freeze(sid)
        if sid in self.history:
            del self.history[sid]

    # backward-compatible alias
    def reset_session_baselines(self, session_id: Optional[str] = None) -> None:
        self.reset_session(session_id)
