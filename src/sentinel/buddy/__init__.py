"""Buddy — decorative ASCII companion (does not help with code).

Hatch is deterministic from user_id. Commands:
  /buddy       show or hatch buddy
  /buddy pet   pet animation
"""

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
]
