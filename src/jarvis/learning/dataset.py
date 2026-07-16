"""Training dataset capture for later fine-tuning.

Records correction pairs as JSONL so they can be exported and used to fine-tune
(or LoRA/QLoRA) a local model later. Each row is a ``(prompt, chosen, rejected)``
preference pair: what the user preferred versus what the model originally said.

Fail-open: a write error is logged, not raised, so capturing never breaks a
reply.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional


class TrainingDataset:
    """Append-only JSONL store of preference pairs."""

    def __init__(self, path: str = "") -> None:
        self.path = path or os.path.join(
            os.path.expanduser("~"), ".jarvis", "training_pairs.jsonl"
        )
        self._lock = threading.Lock()

    def add_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: Optional[str] = None,
    ) -> bool:
        """Append a preference pair. Returns True if written."""
        prompt = (prompt or "").strip()
        chosen = (chosen or "").strip()
        if not prompt or not chosen:
            return False
        row: Dict[str, str] = {"prompt": prompt, "chosen": chosen}
        if rejected:
            row["rejected"] = rejected.strip()
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return True
        except Exception as exc:
            from ..debug import debug_log

            debug_log(f"training pair write failed: {exc}", "learning")
            return False

    def all(self) -> List[Dict[str, str]]:
        """Return all recorded pairs (empty list if none / unreadable)."""
        rows: List[Dict[str, str]] = []
        try:
            if not os.path.exists(self.path):
                return rows
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:
            from ..debug import debug_log

            debug_log(f"training pair read failed: {exc}", "learning")
        return rows
