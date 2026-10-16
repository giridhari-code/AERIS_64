"""Deterministic Buddy hatch + pet — pure companion, no coding help."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from sentinel.buddy.animations import animations_as_text, build_animations
from sentinel.buddy.cosmetics import EYES, HATS, rarity_rank
from sentinel.buddy.mood import MOOD_META, sense_mood
from sentinel.buddy.species import SPECIES, idle_frames

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
    return hashlib.sha256(f"sentinel-buddy-v1:{user_id}".encode()).hexdigest()


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


def _bar(value: Any, width: int = 10) -> str:
    """ASCII progress bar for /buddy stats."""
    try:
        v = max(0, min(int(value), width))
    except (TypeError, ValueError):
        v = 0
    return "█" * v + "░" * (width - v)


def _joker_echo(text: str) -> str:
    """Alternating-case echo — deterministic joker mimicry (no RNG)."""
    t = (text or "").strip()
    if len(t) > 140:
        t = t[:137].rstrip() + "..."
    return "".join(ch.upper() if i % 2 else ch.lower() for i, ch in enumerate(t))


_ANIM_LINES = {
    "wave": "Buddy waves! 👋",
    "happy": "Buddy khush ho gaya! 😊",
    "excited": "Buddy excited — full jump mode! 🤩",
    "sleep": "Buddy ko neend aa rahi hai… 😴",
    "think": "Buddy soch me hai… 🤔",
    "pet": "Buddy ko pyaar mila! 💛",
    "talk": "Buddy baat kar raha hai.",
}
_ANIM_ALIASES = {"dance": "excited", "party": "excited"}


def _anim_token(name: str, line: str | None = None) -> str:
    """Control token: the float plays ``name``, chat shows ``line``."""
    return f"BUDDY_ANIM:{name}\n{line or _ANIM_LINES.get(name, 'Buddy!')}"


def resolve_buddy_reply(reply: str) -> tuple[str, dict]:
    """Map control tokens/prefixes to (chat message, UI meta)."""
    t = (reply or "").strip()
    if t == "BUDDY_ON":
        return "Buddy on — float only; linked to you + AERIS-64.", {"buddy": True, "buddy_ui": "on"}
    if t == "BUDDY_OFF":
        return "Buddy hidden.", {"buddy": True, "buddy_ui": "off"}
    if t == "BUDDY_EGG_UI":
        return (
            "Egg incubating — see Buddy float (hatch is not shown in chat).",
            {"buddy": True, "buddy_egg": True},
        )
    m = re.match(r"^BUDDY_ANIM:([a-z]+)\n?(.*)$", t, flags=re.DOTALL)
    if m:
        name = m.group(1).lower()
        rest = (m.group(2) or "").strip()
        return (rest or _ANIM_LINES.get(name, "Buddy!")), {"buddy": True, "buddy_anim": name}
    return t, {"buddy": True}


def _norm_buddy_text(text: str) -> str:
    """Accept `/buddy ...`, `--buddy ...` and bare `buddy ...` spellings."""
    t = (text or "").strip()
    if t.lower().startswith("--buddy"):
        t = "/" + t[2:]
    if re.match(r"^buddy\s+", t, flags=re.IGNORECASE) and not t.lower().startswith("/buddy"):
        t = "/" + t
    return t


_SUBS = (
    "on|off|enable|disable|hide|show|hatch|status|egg|pet|help|who|me|link|linked|"
    "mimic|joker|mirror|mood|stats|about|aeris|aeris64|dance|party|wave|happy|"
    "sleep|excited|think"
)


def is_buddy_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    t = _norm_buddy_text(text).lower()
    if t in {"/buddy", "/buddy on", "/buddy off", "/buddyon", "/buddyoff"}:
        return True
    if raw.startswith(("/buddy", "--buddy")):
        return t.startswith("/buddy ")
    # bare-word form: only known subcommands ("buddy mood", "buddy on", …)
    return bool(re.match(rf"^/buddy\s+({_SUBS})\b", t))


def handle_buddy_command(
    text: str,
    user_id: str,
    egg: dict | None = None,
    last_text: str | None = None,
    hatch_id: str | None = None,
) -> str | None:
    """Return reply text (or a UI control token) if this is a buddy command, else None.

    Egg incubation UI lives on the **Buddy float only** — chat never gets hatch
    ASCII art (``BUDDY_EGG_UI`` tells the client to refresh the float).
    ``last_text`` powers `/buddy mimic` and `/buddy mood` (previous message);
    ``hatch_id`` pins the companion identity after a guest→account adopt.
    """
    t = _norm_buddy_text(text)
    if not is_buddy_command(t):
        return None
    parts = t.split()
    uid = (user_id or "anonymous").strip() or "anonymous"
    key = (hatch_id or uid).strip() or "anonymous"
    sub = parts[1].lower() if len(parts) >= 2 else "show"
    arg = " ".join(parts[2:]).strip()

    if sub in {"on", "enable"}:
        return "BUDDY_ON"
    if sub in {"off", "hide", "disable"}:
        return "BUDDY_OFF"
    if len(parts) == 1 or sub in {"show", "hatch", "status", "egg"}:
        # Never dump egg art into the chat box — float owns hatching UI.
        if egg is not None and not egg.get("hatched"):
            return "BUDDY_EGG_UI"
        b = hatch_buddy(key)
        shiny = ", shiny" if b.shiny else ""
        return (
            f"{b.name} · {b.species_name} ({b.rarity}{shiny})\n"
            f"Linked to you + AERIS-64 · {b.personality}"
        )
    if sub == "pet":
        return pet_buddy(key)
    if sub in {"mimic", "joker", "mirror"}:
        src = arg or (last_text or "").strip()
        if not arg and src.lower().startswith(("/buddy", "--buddy")):
            src = ""  # don't mimic other buddy commands
        echo = _joker_echo(src)
        if not echo:
            return _anim_token("happy", "Hehe… pehle kuch bolo, phir main nakal karunga! 😜")
        tail = "Joker mode: nakal complete! 😜" if sub == "joker" else "Hehe — bilkul wahi bola! 😜"
        return _anim_token("excited", f"😜 Buddy nakal:\n> {echo}\n{tail}")
    if sub == "mood":
        if not (last_text or "").strip():
            return _anim_token("blink", "Mood: **neutral** 🙂 — pehle kuch bolo, main emotion padhta hoon.")
        label = sense_mood(last_text or "")
        emoji, line, anim = MOOD_META.get(label, MOOD_META["neutral"])
        return _anim_token(anim, f"Mood: **{label}** {emoji} — {line}\n(float pe `{anim}` animation)")
    if sub in {"stats", "stat"}:
        b = hatch_buddy(key)
        return (
            f"**{b.name}** stats (1–10):\n"
            f"  curiosity  `{_bar(b.stats.get('curiosity', 5))}`\n"
            f"  calm       `{_bar(b.stats.get('calm', 5))}`\n"
            f"  loyalty    `{_bar(b.stats.get('loyalty', 5))}`\n"
            f"Personality: {b.personality}"
        )
    if sub in {"who", "me", "link", "linked"}:
        b = hatch_buddy(key)
        pinned = f" (buddy key `{key}`)" if key != uid else ""
        return (
            f"Buddy linked to user `{uid}`{pinned} + product **AERIS-64**.\n"
            f"{b.name} the {b.species_name} ({b.rarity}{', shiny' if b.shiny else ''})\n"
            f"Same login → same egg + buddy on every device."
        )
    if sub in {"about", "aeris", "aeris64"}:
        return (
            "**AERIS-64** ka companion — decorative buddy, AERIS-64 ke saath linked.\n"
            "· account se linked: same login = same egg/buddy har device pe\n"
            "· hatching sirf Buddy float pe — chat box clean rehta hai\n"
            "· chat, streaming, tools, memory, accounts — sab AERIS-64 ke andar\n"
            "· `/buddy help` — saare commands"
        )
    if sub in {"dance", "party", "wave", "happy", "sleep", "excited", "think"}:
        return _anim_token(_ANIM_ALIASES.get(sub, sub))
    if sub in {"help", "?"}:
        return (
            "Buddy commands (float UI — hatch is NOT in the chat box):\n"
            "  /buddy on | --buddy on     — show companion\n"
            "  /buddy off | --buddy off   — hide\n"
            "  /buddy                    — short status (egg progress on float)\n"
            "  /buddy egg                — refresh egg on float\n"
            "  /buddy pet                — pet animation\n"
            "  /buddy mimic [text]       — joker nakal (last message / given text)\n"
            "  /buddy joker [text]       — full joker-mode mimicry\n"
            "  /buddy mood               — aapki emotion padho (last message)\n"
            "  /buddy stats              — curiosity / calm / loyalty\n"
            "  /buddy who | link         — user + AERIS-64 link\n"
            "  /buddy about              — AERIS-64 × buddy\n"
            "  /buddy dance | wave | happy | sleep | excited\n"
            "Same user_id → same buddy forever. Chat messages incubate the egg."
        )
    return "Unknown. Try `/buddy help` or `--buddy on`."
