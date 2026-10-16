"""Buddy — decorative ASCII companion (does not help with code).

Hatch is deterministic from user_id, but a Pokémon-style egg stage now
gates the reveal: every chat message is one incubation step and passive
time also counts (see ``sentinel.buddy.eggs``; ``POST /v1/buddy/step``).
Commands:
  /buddy       show egg status, or the buddy once hatched
  /buddy egg   incubation progress
  /buddy pet   pet animation
"""

from sentinel.buddy.animations import ANIMATION_TYPES, list_animation_types
from sentinel.buddy.eggs import (
    EggStore,
    crack_frames,
    egg_frames,
    egg_status_text,
    required_steps,
)
from sentinel.buddy.engine import (
    Buddy,
    handle_buddy_command,
    hatch_buddy,
    is_buddy_command,
    pet_buddy,
)
from sentinel.buddy.mood import mood_from_text

__all__ = [
    "ANIMATION_TYPES",
    "Buddy",
    "EggStore",
    "crack_frames",
    "egg_frames",
    "egg_status_text",
    "handle_buddy_command",
    "hatch_buddy",
    "is_buddy_command",
    "list_animation_types",
    "mood_from_text",
    "pet_buddy",
    "required_steps",
]
