"""Skill system - load on-demand capability packages from disk.

A skill is a plain Markdown file under ``skills_dir`` (default
``~/.jarvis/skills/``) with a small frontmatter header and a prompt body. Skills
let a small local model "get smarter" without weight changes: when a query
matches a skill's triggers, the skill's prompt fragment is injected into the
system context so the model behaves as if it had that skill.

Skills are optional and never block a normal reply. A skill may reference a
plugin tool (via the ``tool:`` field) so it can pair a behaviour with an action,
but that binding is lazy and only used when the referenced plugin is registered.

Discovery mirrors the plugin system: every ``*.md`` file (excluding those
starting with ``_``) is read and registered. Failures are logged, not fatal.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Dict, List, Optional

from ..debug import debug_log


_FRONTMATTER_RE = re.compile(r"^\s*([A-Za-z_]+):\s*(.+?)\s*$", re.MULTILINE)
_BODY_RE = re.compile(r"^-{3,}\s*(.*)$", re.DOTALL)


class Skill:
    """A single skill: metadata plus the prompt fragment to inject."""

    def __init__(
        self,
        name: str,
        description: str,
        triggers: List[str],
        prompt: str,
        tool: Optional[str] = None,
        source_path: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.triggers = [t.lower() for t in triggers]
        self.prompt = prompt
        self.tool = tool
        self.source_path = source_path
        # camelCase-ish key used for registry lookup.
        self.key = _to_skill_key(name)

    def matches(self, query: str) -> bool:
        q = (query or "").lower()
        return any(t in q for t in self.triggers if t)


def _to_skill_key(name: str) -> str:
    parts = re.split(r"[\s\-_]+", name.strip())
    if not parts:
        return name
    return parts[0].lower() + "".join(p.title() for p in parts[1:])


class SkillManager:
    """Discovers, indexes, and matches skills for a query."""

    def __init__(
        self,
        skills_dir: str = "",
        enabled: bool = True,
    ) -> None:
        self.skills_dir = skills_dir or os.path.join(
            os.path.expanduser("~"), ".jarvis", "skills"
        )
        self.enabled = enabled
        self.skills: Dict[str, Skill] = {}
        self._lock = threading.Lock()

    def discover(self) -> int:
        """Read every ``*.md`` skill file. Returns the count loaded."""
        if not self.enabled:
            return 0
        loaded = 0
        path = self.skills_dir
        if not path or not os.path.isdir(path):
            return 0
        with self._lock:
            self.skills = {}
            for entry in sorted(os.listdir(path)):
                if not entry.endswith(".md") or entry.startswith("_"):
                    continue
                full = os.path.join(path, entry)
                try:
                    skill = _read_skill_file(full)
                    if skill:
                        self.skills[skill.key] = skill
                        loaded += 1
                        debug_log(f"skill loaded: {skill.key}", "skills")
                except Exception as exc:
                    debug_log(f"failed to load skill {entry}: {exc}", "skills")
        return loaded

    def get_relevant_skills(self, query: str) -> List[Skill]:
        if not self.enabled:
            return []
        with self._lock:
            return [s for s in self.skills.values() if s.matches(query)]

    def build_skill_context(self, query: str) -> Optional[str]:
        """Return a ``<Skills>`` prompt block for matching skills, or None."""
        if not self.enabled:
            return None
        matches = self.get_relevant_skills(query)
        if not matches:
            return None
        lines = ["Skills available for this request:"]
        for s in matches:
            lines.append(f"- {s.name}: {s.prompt.strip()}")
        return "\n".join(lines)

    def add_skill(
        self,
        name: str,
        description: str,
        triggers: List[str],
        prompt: str,
        tool: Optional[str] = None,
    ) -> str:
        """Write a new skill file and register it. Returns its path."""
        os.makedirs(self.skills_dir, exist_ok=True)
        key = _to_skill_key(name)
        path = os.path.join(self.skills_dir, f"{key}.md")
        triggers_line = ", ".join(triggers)
        body = f"""name: {name}
description: {description}
triggers: {triggers_line}
"""
        if tool:
            body += f"tool: {tool}\n"
        body += f"---\n{prompt.strip()}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        debug_log(f"skill written: {path}", "skills")
        return path


def _read_skill_file(path: str) -> Optional[Skill]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    meta: Dict[str, str] = {}
    body = text
    m = _BODY_RE.search(text)
    if m:
        head = text[: m.start()]
        body = m.group(1).strip()
        for fm in _FRONTMATTER_RE.finditer(head):
            meta[fm.group(1).lower()] = fm.group(2).strip()
    else:
        # No body separator: treat leading key: lines as frontmatter.
        for fm in _FRONTMATTER_RE.finditer(text):
            meta[fm.group(1).lower()] = fm.group(2).strip()

    name = meta.get("name")
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]
    description = meta.get("description", "")
    triggers_raw = meta.get("triggers", "")
    triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
    tool = meta.get("tool")
    return Skill(
        name=name,
        description=description,
        triggers=triggers,
        prompt=body,
        tool=tool,
        source_path=path,
    )
