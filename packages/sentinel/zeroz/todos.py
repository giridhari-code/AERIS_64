"""Session todos — Claude Code–similar task list."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Status = Literal["pending", "in_progress", "done", "cancelled"]


@dataclass
class TodoItem:
    id: int
    text: str
    status: Status = "pending"


@dataclass
class TodoList:
    items: list[TodoItem] = field(default_factory=list)
    _next_id: int = 1

    def add(self, text: str) -> TodoItem:
        item = TodoItem(id=self._next_id, text=text.strip())
        self._next_id += 1
        self.items.append(item)
        return item

    def set_status(self, item_id: int, status: Status) -> bool:
        for it in self.items:
            if it.id == item_id:
                it.status = status
                return True
        return False

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        lines = []
        for it in self.items:
            mark = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "cancelled": "[-]"}[it.status]
            lines.append(f"{mark} #{it.id} {it.text}")
        return "\n".join(lines)
