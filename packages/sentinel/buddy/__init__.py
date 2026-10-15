"""Buddy — decorative ASCII companion (does not help with code).

Hatch is deterministic from user_id. Commands:
  /buddy       show or hatch buddy
  /buddy pet   pet animation
"""

from sentinel.buddy.animations import list_animation_types, ANIMATION_TYPES
from sentinel.buddy.mood import mood_from_text
from sentinel.buddy.engine import (
    Buddy,
    hatch_buddy,
    pet_buddy,
    handle_buddy_command,
    is_buddy_command,
)

__all__ = [
    "Buddy",
    "hatch_buddy",
    "pet_buddy",
    "handle_buddy_command",
    "is_buddy_command",
    "list_animation_types",
    "ANIMATION_TYPES",
    "mood_from_text",
]
