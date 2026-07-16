"""Tests for the learning subsystem (learning/author.py, store.py, dataset.py).

Verifies intent classification, persistent storage of learned instructions and
skills, training-pair capture, and fail-open behaviour. No live LLM required:
the author's classifier is injected/mocked so tests assert observable outcomes.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jarvis.learning.store import LearnedStore
from jarvis.learning.dataset import TrainingDataset
from jarvis.learning.author import LearningAuthor, Intent


def test_learned_store_remembers_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "learned.json")
        store = LearnedStore(path)
        store.add_instruction("I prefer concise replies in British English.")
        store.add_instruction("Call me by my first name.")
        assert len(store.instructions) == 2
        # Reload from disk simulates a restart.
        reloaded = LearnedStore(path)
        assert len(reloaded.instructions) == 2
        assert "British English" in reloaded.instructions[0]


def test_learned_store_dedupes_identical():
    with tempfile.TemporaryDirectory() as tmp:
        store = LearnedStore(os.path.join(tmp, "learned.json"))
        store.add_instruction("same fact")
        store.add_instruction("same fact")
        assert len(store.instructions) == 1


def test_learned_store_builds_context_block():
    with tempfile.TemporaryDirectory() as tmp:
        store = LearnedStore(os.path.join(tmp, "learned.json"))
        store.add_instruction("Reply as a pirate on weekends.")
        block = store.build_context()
        assert "pirate" in block.lower()
        assert "<Learned>" in block


def test_training_dataset_appends_and_exports_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.jsonl")
        ds = TrainingDataset(path)
        ds.add_pair(prompt="What is 2+2?", chosen="4", rejected="It is 5.")
        ds.add_pair(prompt="Greet me", chosen="Hello!", rejected="Yo.")
        rows = ds.all()
        assert len(rows) == 2
        assert rows[0]["chosen"] == "4"
        # Export round-trips.
        exported = TrainingDataset(path).all()
        assert len(exported) == 2


def test_author_classifies_remember():
    author = LearningAuthor(classifier=lambda text: Intent.REMEMBER)
    intent, payload = author.classify("remember that I drink coffee at 9am")
    assert intent == Intent.REMEMBER
    assert "coffee" in payload["text"]


def test_author_classifies_create_skill():
    author = LearningAuthor(classifier=lambda text: Intent.CREATE_SKILL)
    intent, payload = author.classify("create a skill: Morning Briefing")
    assert intent == Intent.CREATE_SKILL
    assert payload["name"] == "Morning Briefing"


def test_author_classifies_correct():
    author = LearningAuthor(classifier=lambda text: Intent.CORRECT)
    intent, payload = author.classify("no, better: just say the time plainly")
    assert intent == Intent.CORRECT
    assert "time plainly" in payload["correction"]


def test_author_classifies_normal_passthrough():
    author = LearningAuthor(classifier=lambda text: Intent.NORMAL)
    intent, _ = author.classify("what time is it")
    assert intent == Intent.NORMAL


def test_author_handle_remembers_via_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = LearnedStore(os.path.join(tmp, "learned.json"))
        ds = TrainingDataset(os.path.join(tmp, "pairs.jsonl"))
        author = LearningAuthor(
            classifier=lambda t: Intent.REMEMBER, store=store, dataset=ds
        )
        reply = author.handle("remember that I use Vim", query="remember that I use Vim")
        assert reply is not None
        assert any("Vim" in i for i in store.instructions)


def test_author_handle_create_skill_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        skills_dir = os.path.join(tmp, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        store = LearnedStore(os.path.join(tmp, "learned.json"))
        ds = TrainingDataset(os.path.join(tmp, "pairs.jsonl"))
        author = LearningAuthor(
            classifier=lambda t: Intent.CREATE_SKILL,
            store=store,
            dataset=ds,
            skills_dir=skills_dir,
        )
        reply = author.handle(
            "create a skill: Standup Notes that triggers on standup",
            query="create a skill: Standup Notes that triggers on standup",
        )
        assert reply is not None
        import glob
        written = glob.glob(os.path.join(skills_dir, "*.md"))
        assert written, "skill file should be written"


def test_author_handle_correct_appends_training_pair():
    with tempfile.TemporaryDirectory() as tmp:
        store = LearnedStore(os.path.join(tmp, "learned.json"))
        ds = TrainingDataset(os.path.join(tmp, "pairs.jsonl"))
        author = LearningAuthor(
            classifier=lambda t: Intent.CORRECT, store=store, dataset=ds
        )
        author.handle(
            "no, better: say the time only",
            query="what is the time",
            last_reply="The current time is 3pm, a fine hour indeed.",
        )
        assert len(ds.all()) == 1
        assert ds.all()[0]["chosen"] == "say the time only"


def test_author_disabled_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        author = LearningAuthor(
            enabled=False,
            classifier=lambda t: Intent.REMEMBER,
            store=LearnedStore(os.path.join(tmp, "learned.json")),
            dataset=TrainingDataset(os.path.join(tmp, "pairs.jsonl")),
        )
        assert author.handle("remember I like tea", query="remember I like tea") is None
