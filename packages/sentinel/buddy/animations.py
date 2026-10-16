"""Multi-type Buddy ASCII animations — expressions, motion, talk, think, etc."""

from __future__ import annotations

from typing import Any

ANIMATION_TYPES = (
    "idle",
    "blink",
    "float",
    "talk",
    "think",
    "wave",
    "happy",
    "sleep",
    "excited",
    "look_left",
    "look_right",
    "pet",
)


def _pad(lines: list[str], width: int = 11) -> list[str]:
    return [ln.ljust(width)[:width] for ln in lines]


def _eyes(line: str, left: str, right: str | None = None) -> str:
    right = left if right is None else right
    out: list[str] = []
    seen = 0
    for ch in line:
        if ch in "oO.*+-" and seen == 0:
            out.append(left)
            seen = 1
        elif ch in "oO.*+-" and seen == 1:
            out.append(right)
            seen = 2
        else:
            out.append(ch)
    return "".join(out)


def build_animations(base: list[str]) -> dict[str, list[list[str]]]:
    """
    From a 3-line base sprite, build frame lists for each animation type.
    Each value is a list of frames; each frame is list[str] lines.
    """
    b = _pad(list(base))
    while len(b) < 3:
        b.append(" " * 11)
    top, mid, bot = b[0], b[1], b[2]

    def frame(*lines: str) -> list[str]:
        return _pad(list(lines))

    idle = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "o", "o"), bot),
    ]
    blink = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "-", "-"), bot),
        frame(top, mid, bot),
    ]
    # vertical bob via leading spaces on all lines
    float_anim = [
        frame(top, mid, bot),
        frame(" " + top.strip(), " " + mid.strip(), " " + bot.strip()),
        frame(top, mid, bot),
        frame(top.strip(), mid.strip(), bot.strip()),
    ]
    talk = [
        frame(top, mid, bot),
        frame(top, mid, bot.replace(" ", "o", 1) if " " in bot else bot[:-1] + "o"),
        frame(top, mid, bot),
        frame(top, _eyes(mid, "o", "O"), bot),
    ]
    # think: bubble above
    think = [
        frame(top, mid, bot),
        frame("   (.)  ", top, mid, bot)[:3] and [ "   o    ", top, mid] ,
        frame("  (...) ", top, mid),
        frame("   ?    ", top, mid),
    ]
    # fix think to always 3 lines + optional bubble as first of meta — keep 3 body lines
    think = [
        frame(top, mid, bot),
        frame("  .     ", mid, bot),
        frame("  o     ", mid, bot),
        frame("  ...?  ", mid, bot),
        frame(top, mid, bot),
    ]
    wave = [
        frame(top, mid, bot),
        frame(top, mid.rstrip() + "/", bot),
        frame(top, mid, bot),
        frame(top, "\\" + mid.lstrip(), bot),
    ]
    happy = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "^", "^"), bot),
        frame(top, _eyes(mid, "o", "o"), bot.replace("^", "v") if "^" in bot else bot),
        frame(top, _eyes(mid, "^", "^"), bot),
    ]
    sleep = [
        frame(top, _eyes(mid, "-", "-"), bot),
        frame(top, _eyes(mid, "-", "-"), bot + " z"),
        frame(top, _eyes(mid, "-", "-"), bot + " Z"),
        frame(top, _eyes(mid, "-", "-"), bot + " z"),
    ]
    excited = [
        frame(top, mid, bot),
        frame("!" + top.strip(), mid, bot),
        frame(top, _eyes(mid, "O", "O"), bot),
        frame(top, mid, bot),
    ]
    look_left = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "*", "o"), bot),
        frame(top, _eyes(mid, ".", "o"), bot),
        frame(top, mid, bot),
    ]
    look_right = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "o", "*"), bot),
        frame(top, _eyes(mid, "o", "."), bot),
        frame(top, mid, bot),
    ]
    pet = [
        frame(top, mid, bot),
        frame(top, _eyes(mid, "^", "^"), bot),
        frame(top, mid, bot + " ♥"),
        frame(top, _eyes(mid, "^", "^"), "  ♥  ♥  "),
        frame(top, mid, bot),
    ]

    return {
        "idle": idle,
        "blink": blink,
        "float": float_anim,
        "talk": talk,
        "think": think,
        "wave": wave,
        "happy": happy,
        "sleep": sleep,
        "excited": excited,
        "look_left": look_left,
        "look_right": look_right,
        "pet": pet,
    }


def animations_as_text(anims: dict[str, list[list[str]]]) -> dict[str, list[str]]:
    """JSON-friendly: each anim → list of frame strings (joined lines)."""
    out: dict[str, list[str]] = {}
    for name, frames in anims.items():
        out[name] = ["\n".join(fr) for fr in frames]
    return out


def list_animation_types() -> list[dict[str, Any]]:
    return [
        {"id": "idle", "desc": "Gentle presence"},
        {"id": "blink", "desc": "Eye blink"},
        {"id": "float", "desc": "Hover up/down"},
        {"id": "talk", "desc": "Talking mouth"},
        {"id": "think", "desc": "Thinking dots"},
        {"id": "wave", "desc": "Wave hello"},
        {"id": "happy", "desc": "Happy eyes"},
        {"id": "sleep", "desc": "Sleepy Zzz"},
        {"id": "excited", "desc": "Excited"},
        {"id": "look_left", "desc": "Look left"},
        {"id": "look_right", "desc": "Look right"},
        {"id": "pet", "desc": "Hearts when petted"},
    ]
