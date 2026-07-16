"""Persistent store for learned instructions.

Learned instructions are short, durable facts or behavioural preferences the
user has taught Jarvis through chat (e.g. "reply in British English", "call me
by my first name"). They are persisted as a flat JSON file so they stay
transparent and editable, and injected into the system prompt as a ``<Learned>``
block when relevant.

Storage is fail-open: any read/write error is logged and the store degrades to
an in-memory, empty state rather than breaking a reply.
"""

from __future__ import annotations

import json
import os
import threading
from typing import List, Optional


class LearnedStore:
    """Loads/saves learned instructions to a JSON file."""

    def __init__(self, path: str = "") -> None:
        self.path = path or os.path.join(
            os.path.expanduser("~"), ".jarvis", "learned.json"
        )
        self.instructions: List[str] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("instructions"), list):
                    self.instructions = [str(i) for i in data["instructions"]]
        except Exception as exc:
            from ..debug import debug_log

            debug_log(f"learned store load failed: {exc}", "learning")
            self.instructions = []

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"instructions": self.instructions}, f, indent=2)
        except Exception as exc:
            from ..debug import debug_log

            debug_log(f"learned store save failed: {exc}", "learning")

    def add_instruction(self, text: str) -> bool:
        """Add an instruction if not already present. Returns True if added."""
        text = (text or "").strip()
        if not text:
            return False
        with self._lock:
            if text in self.instructions:
                return False
            self.instructions.append(text)
            self._save()
            return True

    def build_context(self) -> Optional[str]:
        """Return a ``<Learned>`` prompt block, or None when empty."""
        with self._lock:
            if not self.instructions:
                return None
            lines = ["<Learned>"]
            lines.append("Things the user has taught you (honour these):")
            for i in self.instructions:
                lines.append(f"- {i}")
            return "\n".join(lines)
