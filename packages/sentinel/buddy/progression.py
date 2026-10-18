"""Buddy progression — care, streak, evolution, cosmetics unlocks, soft memory."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

STAGE_THRESHOLDS = (0, 40, 120)
UNLOCK_AT = {
    15: ["flower", "bow"],
    40: ["halo", "propeller"],
    80: ["crown"],
}


def _day_key(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts or time.time()))


@dataclass
class Progression:
    user_id: str
    care: int = 0
    streak: int = 0
    last_day: str = ""
    stage: int = 0
    unlocked: list[str] = field(default_factory=list)
    memory_on: bool = True
    last_mood: str = "idle"
    last_pet_at: float = 0.0
    last_checkin: str = ""
    equipped_hat: str = ""
    equipped_eye: str = ""
    total_pets: int = 0
    total_messages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public(self) -> dict[str, Any]:
        next_stage = None
        if self.stage < 2:
            next_stage = {
                "stage": self.stage + 1,
                "need": STAGE_THRESHOLDS[self.stage + 1],
                "have": self.care,
                "remaining": max(0, STAGE_THRESHOLDS[self.stage + 1] - self.care),
            }
        names = ("hatch", "evolved", "apex")
        return {
            "care": self.care,
            "streak": self.streak,
            "stage": self.stage,
            "stage_name": names[min(2, self.stage)],
            "unlocked": list(self.unlocked),
            "memory_on": self.memory_on,
            "last_mood": self.last_mood if self.memory_on else "",
            "last_pet_at": self.last_pet_at if self.memory_on else 0,
            "equipped_hat": self.equipped_hat,
            "equipped_eye": self.equipped_eye,
            "total_pets": self.total_pets,
            "total_messages": self.total_messages,
            "next_stage": next_stage,
            "checkin": self.last_checkin,
        }


class ProgressionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        env = os.environ.get("NEUROFIELD_BUDDY_PROGRESS_PATH") or os.environ.get(
            "AERIS_BUDDY_PROGRESS_PATH"
        )
        self.path = Path(path or env or "artifacts/buddy/progression.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, user_id: str) -> Progression:
        uid = (user_id or "anonymous").strip() or "anonymous"
        raw = self._load().get(uid) or {}
        return Progression(
            user_id=uid,
            care=int(raw.get("care") or 0),
            streak=int(raw.get("streak") or 0),
            last_day=str(raw.get("last_day") or ""),
            stage=int(raw.get("stage") or 0),
            unlocked=list(raw.get("unlocked") or []),
            memory_on=bool(raw.get("memory_on", True)),
            last_mood=str(raw.get("last_mood") or "idle"),
            last_pet_at=float(raw.get("last_pet_at") or 0),
            last_checkin=str(raw.get("last_checkin") or ""),
            equipped_hat=str(raw.get("equipped_hat") or ""),
            equipped_eye=str(raw.get("equipped_eye") or ""),
            total_pets=int(raw.get("total_pets") or 0),
            total_messages=int(raw.get("total_messages") or 0),
        )

    def put(self, prog: Progression) -> None:
        data = self._load()
        data[prog.user_id] = prog.to_dict()
        self._save(data)

    def _recompute_stage(self, prog: Progression) -> None:
        stage = 0
        for i, th in enumerate(STAGE_THRESHOLDS):
            if prog.care >= th:
                stage = i
        prog.stage = stage
        for need, items in UNLOCK_AT.items():
            if prog.care >= need:
                for it in items:
                    if it not in prog.unlocked:
                        prog.unlocked.append(it)

    def touch_day(self, user_id: str) -> Progression:
        prog = self.get(user_id)
        today = _day_key()
        if prog.last_day == today:
            return prog
        if prog.last_day:
            try:
                prev = time.mktime(time.strptime(prog.last_day, "%Y-%m-%d"))
                if _day_key(prev + 86400) == today:
                    prog.streak = max(1, prog.streak + 1)
                else:
                    prog.streak = 1
            except Exception:
                prog.streak = 1
        else:
            prog.streak = 1
        prog.last_day = today
        prog.care += 2
        self._recompute_stage(prog)
        self.put(prog)
        return prog

    def add_care(self, user_id: str, amount: int, *, mood: str = "") -> Progression:
        prog = self.touch_day(user_id)
        prog.care += max(0, int(amount))
        if mood and prog.memory_on:
            prog.last_mood = mood
        self._recompute_stage(prog)
        self.put(prog)
        return prog

    def pet(self, user_id: str) -> Progression:
        prog = self.add_care(user_id, 3, mood="pet")
        prog.total_pets += 1
        prog.last_pet_at = time.time()
        self.put(prog)
        return prog

    def message_step(self, user_id: str, mood: str = "") -> Progression:
        prog = self.add_care(user_id, 1, mood=mood)
        prog.total_messages += 1
        self.put(prog)
        return prog

    def set_memory(self, user_id: str, on: bool) -> Progression:
        prog = self.get(user_id)
        prog.memory_on = bool(on)
        if not on:
            prog.last_mood = ""
            prog.last_pet_at = 0.0
        self.put(prog)
        return prog

    def equip(self, user_id: str, *, hat: str = "", eye: str = "") -> Progression:
        prog = self.get(user_id)
        if hat and (hat in prog.unlocked or hat in {"none", "cap", "beanie"}):
            prog.equipped_hat = hat
        if eye:
            prog.equipped_eye = eye
        self.put(prog)
        return prog

    def checkin_line(self, user_id: str) -> str:
        prog = self.touch_day(user_id)
        hour = time.localtime().tm_hour
        if 5 <= hour < 12:
            line = "Good morning — stretch & sip water."
        elif 12 <= hour < 17:
            line = "Afternoon check-in — you're doing fine."
        elif 17 <= hour < 22:
            line = "Evening calm — one deep breath."
        else:
            line = "Late night — rest when you can."
        names = ("hatch", "evolved", "apex")
        if prog.streak >= 3:
            line += f" · {prog.streak} day streak"
        line += f" · {names[min(2, prog.stage)]}"
        prog.last_checkin = line
        self.put(prog)
        return line


def evolve_sprite(lines: list[str], stage: int) -> list[str]:
    if stage <= 0 or not lines:
        return list(lines)
    out = list(lines)
    if stage >= 1 and len(out) >= 1 and "*" not in out[0]:
        out[0] = out[0].rstrip() + " *"
    if stage >= 2 and len(out) >= 1:
        out[0] = "·" + out[0].strip() + "·"
    return out


def share_card_text(buddy_public: dict[str, Any], prog: Progression) -> str:
    name = buddy_public.get("name") or "Buddy"
    species = buddy_public.get("species_name") or "Companion"
    rarity = buddy_public.get("rarity") or "common"
    shiny = " shiny" if buddy_public.get("shiny") else ""
    stage = ("hatch", "evolved", "apex")[min(2, prog.stage)]
    sprite = "\n".join(buddy_public.get("sprite_lines") or [])
    return (
        f"AERIS-64 Buddy\n{name} the {species}{shiny}\n"
        f"rarity: {rarity} · stage: {stage} · streak: {prog.streak}\n"
        f"{sprite}\nThink Smarter. Do More."
    )


_STORE: Optional[ProgressionStore] = None


def get_progression_store() -> ProgressionStore:
    global _STORE
    if _STORE is None:
        _STORE = ProgressionStore()
    return _STORE
