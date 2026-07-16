"""Chat-driven authoring: turn a user utterance into a learned skill or fact.

The author sits in front of normal reply generation. When the user says
something that looks like a teaching moment, it is intercepted, handled, and a
short confirmation is returned instead of a full LLM reply. Otherwise the turn
is marked NORMAL and proceeds as usual.

Explicit triggers (cheap, deterministic regex, no model needed):
- "remember that ..."            -> store a learned instruction
- "create a skill: <name> ..."  -> write a skill file
- "no, better: ..."             -> capture a training correction pair

Implicit corrections (Phase 2): when there is a previous reply and the user's
message reads as feedback on it (e.g. "actually, ...", "you're wrong, ...",
"I meant ...", "shorter next time") without being a fresh question, it is
captured as a correction pair (chosen = the user's phrasing, rejected = the
prior reply). This is gated so it never fires on a new question and never
without a prior reply, keeping false positives low on a small model.

An optional injected ``classifier`` (e.g. the small router model) may upgrade a
NORMAL turn to a teaching intent, but the explicit/implicit regex signals win
for their respective patterns so they never depend on the model. All paths are
fail-open.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Optional, Tuple

from .store import LearnedStore
from .dataset import TrainingDataset


class Intent(str, Enum):
    NORMAL = "normal"
    REMEMBER = "remember"
    CREATE_SKILL = "create_skill"
    CORRECT = "correct"


_REMEMBER_RE = re.compile(r"^\s*remember\s+(?:that\s+)?(.+)$", re.IGNORECASE)
_CREATE_RE = re.compile(
    r"^\s*create\s+(?:a\s+)?skill\s*:\s*(.+?)(?:\s+that\s+triggers\s+on\s+(.+))?$",
    re.IGNORECASE,
)
_CORRECT_RE = re.compile(
    r"^\s*(?:no|wrong|not\s+quite)[,:\s]*(?:better\s*)?[:\-]?\s*(.+)$",
    re.IGNORECASE,
)

# Implicit-correction signals: the user is reacting to the prior reply.
_IMPLICIT_CORRECT_RE = re.compile(
    r"""
    ^\s*(
        actually[,:\s]                                   # "actually, ..."
      | (?:you\s+(?:are|re|were)\s+wrong)               # "you're wrong"
      | that\s+is\s+wrong                               # "that is wrong"
      | i\s+meant\b                                     # "I meant ..."
      | instead[,:\s]                                    # "instead, ..."
      | rather[,:\s]                                     # "rather, ..."
      | (?:you\s+should\s+have)                         # "you should have ..."
      | (?:be|keep\s+it|make\s+it)\s+(?:more\s+)?(?:short|brief|concise)  # "be shorter"
      | more\s+(?:short|brief|concise)                  # "more concise"
      | shorter\b                                       # "shorter next time"
      | (?:too\s+(?:long|verbose|wordy))                # "too long"
      | no[,:\s]+(?=\S)                                 # "no, ..." (has content after)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Markers that the utterance is actually a new question, not feedback.
_QUESTION_RE = re.compile(
    r"\?$|^\s*(?:what|when|where|who|whom|which|why|how|can|could|should|"
    r"would|is|are|was|were|do|does|did|will|have|has|am)\b",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    return bool(_QUESTION_RE.search(text or ""))


def _regex_intent(text: str, last_reply: str = "") -> Intent:
    """Cheap, deterministic intent pre-check before any model call."""
    if _REMEMBER_RE.match(text):
        return Intent.REMEMBER
    if _CREATE_RE.match(text):
        return Intent.CREATE_SKILL
    if _CORRECT_RE.match(text):
        return Intent.CORRECT
    # Implicit correction: only when there is a prior reply to react to, and the
    # utterance is feedback rather than a fresh question.
    if last_reply and _IMPLICIT_CORRECT_RE.match(text) and not _looks_like_question(text):
        return Intent.CORRECT
    return Intent.NORMAL


class LearningAuthor:
    """Intercept teaching utterances and persist them."""

    def __init__(
        self,
        enabled: bool = True,
        implicit_corrections_enabled: bool = True,
        classifier: Optional[Callable[[str], Intent]] = None,
        store: Optional[LearnedStore] = None,
        dataset: Optional[TrainingDataset] = None,
        skills_dir: str = "",
    ) -> None:
        self.enabled = enabled
        self.implicit_corrections_enabled = implicit_corrections_enabled
        self.classifier = classifier
        self.store = store or LearnedStore()
        self.dataset = dataset or TrainingDataset()
        self.skills_dir = skills_dir or _default_skills_dir()

    def classify(self, text: str, last_reply: str = "") -> Tuple[Intent, dict]:
        """Return (intent, payload). Payload carries extracted fields."""
        text = (text or "").strip()
        intent = _regex_intent(
            text, last_reply if self.implicit_corrections_enabled else ""
        )
        # A model classifier can override NORMAL when it is supplied; regex wins
        # for the explicit/implicit triggers so they never depend on the model.
        if intent == Intent.NORMAL and self.classifier is not None:
            try:
                intent = self.classifier(text) or Intent.NORMAL
            except Exception:
                intent = Intent.NORMAL
        payload = self._extract(intent, text)
        return intent, payload

    def _extract(self, intent: Intent, text: str) -> dict:
        if intent == Intent.REMEMBER:
            m = _REMEMBER_RE.match(text)
            return {"text": m.group(1).strip() if m else text}
        if intent == Intent.CREATE_SKILL:
            m = _CREATE_RE.match(text)
            if not m:
                return {"name": "", "triggers": []}
            name = m.group(1).strip()
            triggers = [t.strip() for t in (m.group(2) or "").split(",") if t.strip()]
            return {"name": name, "triggers": triggers}
        if intent == Intent.CORRECT:
            m = _CORRECT_RE.match(text)
            correction = m.group(1).strip() if m else text
            # Implicit corrections often start with a feedback lead-in
            # ("actually,", "instead,"); strip it so the chosen text is the
            # substance of the correction.
            correction = _strip_correction_lead_in(correction)
            return {"correction": correction}
        return {}

    def handle(
        self,
        text: str,
        query: str = "",
        last_reply: str = "",
    ) -> Optional[str]:
        """Handle a teaching utterance. Returns a confirmation, or None to
        let the turn proceed as a normal reply."""
        if not self.enabled:
            return None
        intent, payload = self.classify(text, last_reply=last_reply)
        if intent == Intent.REMEMBER:
            added = self.store.add_instruction(payload.get("text", ""))
            return (
                "Noted. I will remember that."
                if added
                else "I already had that one noted."
            )
        if intent == Intent.CREATE_SKILL:
            name = payload.get("name") or "Untitled Skill"
            triggers = payload.get("triggers") or [name]
            from ..skills.manager import SkillManager

            mgr = SkillManager(skills_dir=self.skills_dir)
            mgr.add_skill(
                name=name,
                description=f"User-created skill: {name}",
                triggers=triggers,
                prompt=f"Apply the '{name}' behaviour the user defined.",
            )
            return f"Created the skill '{name}'. I will use it when you mention {', '.join(triggers)}."
        if intent == Intent.CORRECT:
            self.dataset.add_pair(
                prompt=query or text,
                chosen=payload.get("correction", ""),
                rejected=last_reply or None,
            )
            return "Got it, I will do better next time."
        return None


_LEAD_IN_RE = re.compile(
    r"^\s*(?:actually|instead|rather|no)[,:\s]+", re.IGNORECASE
)


def _strip_correction_lead_in(text: str) -> str:
    stripped = _LEAD_IN_RE.sub("", text).strip()
    return stripped or text


def _default_skills_dir() -> str:
    import os

    return os.path.join(os.path.expanduser("~"), ".jarvis", "skills")
