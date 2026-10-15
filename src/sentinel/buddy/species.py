"""18 buddy species — clear names, fixed ASCII sprites (~5 lines)."""

from __future__ import annotations

from typing import TypedDict


class SpeciesDef(TypedDict):
    id: str
    name: str
    sprite: list[str]  # base body lines (eyes/hat overlaid later)


# Order is stable; index chosen deterministically from user hash.
SPECIES: list[SpeciesDef] = [
    {
        "id": "pebblecrab",
        "name": "Pebblecrab",
        "sprite": [
            "  (o o)  ",
            " <(   )> ",
            "  ^^ ^^  ",
        ],
    },
    {
        "id": "nebulynx",
        "name": "Nebulynx",
        "sprite": [
            "  /\\_/\\  ",
            " ( . . ) ",
            "  > ^ <  ",
        ],
    },
    {
        "id": "capybara",
        "name": "Capybara",
        "sprite": [
            "  (---)  ",
            " ( o o ) ",
            "  \\___/  ",
        ],
    },
    {
        "id": "duck",
        "name": "Duck",
        "sprite": [
            "  __(o)> ",
            " (  =  ) ",
            "  ^^ ^^  ",
        ],
    },
    {
        "id": "cat",
        "name": "Cat",
        "sprite": [
            "  /\\_/\\  ",
            " ( o.o ) ",
            "  > ^ <  ",
        ],
    },
    {
        "id": "dragon",
        "name": "Dragon",
        "sprite": [
            "  /^_^\\  ",
            " ( o o ) ",
            " /|   |\\ ",
        ],
    },
    {
        "id": "blobbit",
        "name": "Blobbit",
        "sprite": [
            "  .---.  ",
            " ( o o ) ",
            "  '---'  ",
        ],
    },
    {
        "id": "starfinch",
        "name": "Starfinch",
        "sprite": [
            "   \\ /   ",
            "  (o o)  ",
            "  / ^ \\  ",
        ],
    },
    {
        "id": "mossgoat",
        "name": "Mossgoat",
        "sprite": [
            "  (\\Y/)  ",
            " ( o o ) ",
            "  \"   \"  ",
        ],
    },
    {
        "id": "pixkit",
        "name": "Pixkit",
        "sprite": [
            "  .\" \".  ",
            " ( o o ) ",
            "  \\   /  ",
        ],
    },
    {
        "id": "emberpup",
        "name": "Emberpup",
        "sprite": [
            "  /^ ^\\  ",
            " ( o o ) ",
            "  \\ U /  ",
        ],
    },
    {
        "id": "glintbug",
        "name": "Glintbug",
        "sprite": [
            "   \\o/   ",
            "  (   )  ",
            "   ^ ^   ",
        ],
    },
    {
        "id": "tidalfox",
        "name": "Tidalfox",
        "sprite": [
            "  /\\_/\\~ ",
            " ( o o ) ",
            "  -\" \"-  ",
        ],
    },
    {
        "id": "quartzmoth",
        "name": "Quartzmoth",
        "sprite": [
            "  \\   /  ",
            "  (o o)  ",
            "   \\_/   ",
        ],
    },
    {
        "id": "sproutling",
        "name": "Sproutling",
        "sprite": [
            "   \\|/   ",
            "  ( o )  ",
            "   |||   ",
        ],
    },
    {
        "id": "cloudsnail",
        "name": "Cloudsnail",
        "sprite": [
            "   ___   ",
            "  (o o)@ ",
            "  /___\\  ",
        ],
    },
    {
        "id": "runeowl",
        "name": "Runeowl",
        "sprite": [
            "  {o,o}  ",
            "  /)_)   ",
            "   \" \"   ",
        ],
    },
    {
        "id": "voltnewt",
        "name": "Voltnewt",
        "sprite": [
            "  (\\_/)  ",
            "  (o o)  ",
            "  /| |\\  ",
        ],
    },
]

assert len(SPECIES) == 18

# Idle frames: subtle second frame per species (blink / sway)
def idle_frames(species_id: str, base: list[str]) -> list[list[str]]:
    """Return [frame0, frame1] for simple idle animation."""
    f0 = list(base)
    f1 = list(base)
    # blink: replace first 'o' on middle line with '-'
    if len(f1) >= 2:
        mid = f1[1]
        f1[1] = mid.replace("o", "-", 1).replace("O", "-", 1)
    return [f0, f1]
