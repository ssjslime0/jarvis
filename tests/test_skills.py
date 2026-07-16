"""Tests for the skill system (skills/manager.py).

Skills are prompt-context packages the small model loads on demand. A skill is
a Markdown file with YAML-ish frontmatter (name, description, triggers, optional
tool) plus a prompt body. These tests verify discovery, relevance matching, and
context assembly without requiring a live LLM.
"""

import sys
import os
import tempfile
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jarvis.skills.manager import Skill, SkillManager


_SKILL_A = textwrap.dedent(
    """
    name: Git Helper
    description: Explains git commands and workflows
    triggers: git, commit, branch, rebase, merge
    ---
    When the user asks about git, prefer concrete command snippets and explain
    the trade-offs of rebasing versus merging in one short sentence.
    """
).strip()

_SKILL_B = textwrap.dedent(
    """
    name: Calm Down
    description: Helps the user relax and de-stress
    triggers: stress, anxious, overwhelmed, breathe
    ---
    For wellbeing topics, suggest one small realistic step and stay calm.
    """
).strip()


def _write(tmp, name, body):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    return path


def test_skill_parses_frontmatter_and_body():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        assert "gitHelper" in mgr.skills, mgr.skills.keys()
        sk = mgr.skills["gitHelper"]
        assert sk.name == "Git Helper"
        assert "git" in sk.triggers
        assert "rebase" in sk.prompt.lower()


def test_discover_skips_non_md_and_files_starting_with_underscore():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        _write(tmp, "README.txt", "ignore me")
        _write(tmp, "_draft.md", _SKILL_B)
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        assert set(mgr.skills.keys()) == {"gitHelper"}


def test_get_relevant_skills_matches_trigger_keywords():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        _write(tmp, "calm.md", _SKILL_B)
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        matches = mgr.get_relevant_skills("how do I rebase this branch")
        names = {s.name for s in matches}
        assert "Git Helper" in names
        assert "Calm Down" not in names


def test_get_relevant_skills_empty_when_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        assert mgr.get_relevant_skills("what is the weather today") == []


def test_build_skill_context_assembles_block():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        ctx = mgr.build_skill_context("explain git commit")
        assert "Git Helper" in ctx
        assert "git" in ctx.lower()


def test_build_skill_context_none_when_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "git.md", _SKILL_A)
        mgr = SkillManager(skills_dir=tmp, enabled=False)
        mgr.discover()
        assert mgr.build_skill_context("explain git commit") is None


def test_add_skill_writes_file_and_registers():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillManager(skills_dir=tmp)
        path = mgr.add_skill(
            name="Standup Prep",
            description="Helps draft a daily standup",
            triggers=["standup", "daily update"],
            prompt="Summarise yesterday, today, and blockers concisely.",
        )
        assert os.path.exists(path)
        mgr.discover()
        assert "standupPrep" in mgr.skills
        assert "standup" in mgr.skills["standupPrep"].triggers


def test_empty_dir_yields_no_skills():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillManager(skills_dir=tmp)
        mgr.discover()
        assert mgr.skills == {}
        assert mgr.build_skill_context("anything") is None
