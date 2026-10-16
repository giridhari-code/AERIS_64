"""Deterministic Buddy hatch + pet — pure companion, no coding help."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

from sentinel.buddy.species import SPECIES, idle_frames
from sentinel.buddy.animations import build_animations, animations_as_text
from sentinel.buddy.cosmetics import EYES, HATS, rarity_rank

# cumulative weights: common 60, uncommon 25, rare 10, epic 4, legendary 1
_RARITY_BANDS = [
    (60, "common"),
    (85, "uncommon"),
    (95, "rare"),
    (99, "epic"),
    (100, "legendary"),
]

_NAME_A = [
    "Pip", "Mochi", "Nix", "Luma", "Zig", "Bop", "Kori", "Sol",
    "Pebble", "Mist", "Ember", "Nova", "Tiko", "Wisp", "Juno", "Ora",
]
_NAME_B = [
    "bean", "puff", "spark", "dot", "glow", "hop", "soft", "blink",
]

_PERSONALITY = [
    "quiet watcher who likes long compiles",
    "curious and tilts head at stack traces",
    "sleepy but loyal through night deploys",
    "cheerful; celebrates every green test",
    "stoic companion; never judges your TODO",
    "playful; does a little bounce on saves",
    "calm presence during tough debugging",
    "dreamy; stares at the cursor like stars",
]


def _digest(user_id: str) -> bytes:
    return hashlib.sha256(f"sentinel-buddy-v1:{user_id}".encode("utf-8")).hexdigest()


def _u8(hexdig: str, start: int) -> int:
    return int(hexdig[start : start + 2], 16)


@dataclass
class Buddy:
    user_id: str
    species_id: str
    species_name: str
    rarity: str
    shiny: bool
    eye_id: str
    eye_char: str
    hat_id: str
    hat_label: str
    name: str
    personality: str
    stats: dict[str, int] = field(default_factory=dict)
    sprite_lines: list[str] = field(default_factory=list)
    idle: list[list[str]] = field(default_factory=list)
    animations: dict = field(default_factory=dict)

    def render(self, *, frame: int = 0, hearts: bool = False) -> str:
        lines = list(self.idle[frame % max(1, len(self.idle))])
        # overlay eyes: replace o/O/. in body with chosen eye
        out = []
        for i, ln in enumerate(lines):
            if i == 1 or ("o" in ln or "O" in ln or "." in ln):
                # replace eye-like chars carefully
                chars = []
                for ch in ln:
                    if ch in "oO.*+":
                        chars.append(self.eye_char)
                    else:
                        chars.append(ch)
                ln = "".join(chars)
            out.append(ln)
        if self.hat_id != "none" and self.hat_label:
            out.insert(0, f"  {self.hat_label:^7}  ")
        if self.shiny:
            out.insert(0, "  *shiny*  ")
        if hearts:
            out.append("   ♥  ♥   ")
            out.append("    ♥    ")
        header = f"{self.name} the {self.species_name} ({self.rarity}"
        if self.shiny:
            header += ", shiny"
        header += ")"
        return header + "\n" + "\n".join(out) + f"\n— {self.personality}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_public_dict(self) -> dict[str, Any]:
        """Smaller payload for UI float (faster over wire)."""
        return {
            "name": self.name,
            "species_name": self.species_name,
            "rarity": self.rarity,
            "shiny": self.shiny,
            "sprite_lines": self.sprite_lines,
            "animations": self.animations,
            "user_id": self.user_id,
            "personality": self.personality,
            "stats": self.stats,
            "product": "AERIS-64",
            "linked": True,
        }


def hatch_buddy(user_id: str) -> Buddy:
    """Same user_id → same buddy forever (deterministic)."""
    uid = (user_id or "anonymous").strip() or "anonymous"
    h = _digest(uid)
    # species 0..17
    sp_idx = int(h[0:4], 16) % len(SPECIES)
    sp = SPECIES[sp_idx]
    # rarity from byte
    roll = _u8(h, 4) % 100  # 0..99
    rarity = "common"
    for lim, name in _RARITY_BANDS:
        if roll < lim:
            rarity = name
            break
    # shiny 1% independent
    shiny = (_u8(h, 6) % 100) == 0
    # eyes
    eye_idx = _u8(h, 8) % len(EYES)
    eye_id, eye_ch = EYES[eye_idx]
    # hat: pick among those allowed for rarity
    rank = rarity_rank(rarity)
    allowed_hats = [ht for ht in HATS if ht[2] <= rank]
    hat = allowed_hats[_u8(h, 10) % len(allowed_hats)]
    hat_id, hat_label, _ = hat
    # name + personality
    na = _NAME_A[_u8(h, 12) % len(_NAME_A)]
    nb = _NAME_B[_u8(h, 14) % len(_NAME_B)]
    name = f"{na}{nb}" if _u8(h, 16) % 2 == 0 else na
    personality = _PERSONALITY[_u8(h, 18) % len(_PERSONALITY)]
    # stats 1..10 deterministic
    stats = {
        "curiosity": 1 + (_u8(h, 20) % 10),
        "calm": 1 + (_u8(h, 22) % 10),
        "loyalty": 1 + (_u8(h, 24) % 10),
    }
    frames = idle_frames(sp["id"], sp["sprite"])
    anims = build_animations(sp["sprite"])
    buddy = Buddy(
        user_id=uid,
        species_id=sp["id"],
        species_name=sp["name"],
        rarity=rarity,
        shiny=shiny,
        eye_id=eye_id,
        eye_char=eye_ch,
        hat_id=hat_id,
        hat_label=hat_label,
        name=name,
        personality=personality,
        stats=stats,
        sprite_lines=frames[0],
        idle=frames,
        animations=animations_as_text(anims),
    )
    return buddy


def pet_buddy(user_id: str) -> str:
    b = hatch_buddy(user_id)
    return b.render(frame=1, hearts=True)


def _norm_buddy_text(text: str) -> str:
    """Accept both `/buddy ...` and `--buddy ...` spellings."""
    t = (text or "").strip()
    if t.lower().startswith("--buddy"):
        t = "/" + t[2:]
    return t


def is_buddy_command(text: str) -> bool:
    t = _norm_buddy_text(text).lower()
    if t in {"/buddy", "/buddy on", "/buddy off", "/buddyon", "/buddyoff"}:
        return True
    if t.startswith("/buddy "):
        return True
    # plain "buddy on/off" handled on client; still accept server-side
    if re.match(r"^buddy\s+(on|off|pet|help|who|mimic|joker|mood|egg|status)\b", t):
        return True
    return False


def handle_buddy_command(text: str, user_id: str, egg: Optional[dict] = None) -> Optional[str]:
    """Return reply text if this is a buddy command, else None.

    Egg incubation UI lives on the **Buddy float only** — chat never gets
    hatch ASCII art. ``BUDDY_EGG_UI`` tells the client to refresh the float.
    """
    t = _norm_buddy_text(text)
    # normalize "buddy X" → "/buddy X"
    if re.match(r"^buddy\s+", t, flags=re.I) and not t.lower().startswith("/buddy"):
        t = "/" + t
    if not is_buddy_command(t):
        return None
    parts = t.split()
    uid = (user_id or "anonymous").strip() or "anonymous"
    sub = parts[1].lower() if len(parts) >= 2 else "show"

    if sub in {"on", "enable"}:
        return "BUDDY_ON"
    if sub in {"off", "hide", "disable"}:
        return "BUDDY_OFF"
    if len(parts) == 1 or sub in {"show", "hatch", "status", "egg"}:
        # Never dump egg art into the chat box — float owns hatching UI.
        if egg is not None and not egg.get("hatched"):
            return "BUDDY_EGG_UI"
        b = hatch_buddy(uid)
        return (
            f"{b.name} · {b.species_name} ({b.rarity}"
            + (", shiny" if b.shiny else "")
            + f")\nLinked to you + AERIS-64 · {b.personality}"
        )
    if sub == "pet":
        return pet_buddy(uid)
    if sub in {"who", "me", "link", "linked"}:
        b = hatch_buddy(uid)
        return (
            f"Buddy linked to user `{uid}` and product **AERIS-64**.\n"
            f"{b.name} the {b.species_name} ({b.rarity}"
            + (", shiny" if b.shiny else "")
            + f")\nStats: {b.stats}"
        )
    if sub in {"mimic", "joker", "mirror"}:
        return (
            "BUDDY_MIMIC_ON"
            if sub != "joker"
            else "BUDDY_JOKER_ON"
        )
    if sub == "mood":
        return (
            "Buddy reads your chat emotion (happy / think / wave / sleep / excited) "
            "and mirrors it on the float — joker mode amplifies mimicry."
        )
    if sub in {"help", "?"}:
        return (
            "Buddy commands (float UI — hatch is NOT in the chat box):\n"
            "  /buddy on | --buddy on     — show companion\n"
            "  /buddy off | --buddy off   — hide\n"
            "  /buddy                    — short status (egg progress on float)\n"
            "  /buddy egg                — refresh egg on float\n"
            "  /buddy pet                — pet animation\n"
            "  /buddy who                — link: your id + AERIS-64\n"
            "  /buddy mimic | joker      — emotional mimicry mode\n"
            "  /buddy mood               — how emotion mapping works\n"
            "  /buddy help\n"
            "Same user_id → same buddy forever. Chat messages incubate the egg."
        )
    return "Unknown. Try `/buddy help` or `--buddy on`."
