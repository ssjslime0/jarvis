"""Real end-to-end smoke test of the chat-driven learning + skills system.

Drives the actual engine against the local Ollama gemma2:2b model and the real
authoring code paths. Isolated temp dirs are used for skill/learned/training
state so the user's real ~/.jarvis profile is not polluted.
"""

import sys
import os
import types
import json
import tempfile
import shutil
import io
from dataclasses import replace


# Mirror the daemon's Windows UTF-8 stdout setup so emoji prints behave as in
# production (the daemon does this; importing the engine alone does not).
if sys.platform == "win32" and not getattr(sys, "frozen", False):
    try:
        if hasattr(sys.stdout, "buffer") and hasattr(sys.stdout.buffer, "write"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer") and hasattr(sys.stderr.buffer, "write"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Stubs for optional deps not installed here.
sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
_mcp = types.ModuleType("mcp")
_mcp.ClientSession = object
_mc = types.ModuleType("mcp.client")
_mc.stdio_client = lambda *a, **k: None
_mc.StdioServerParameters = object
_mcs = types.ModuleType("mcp.client.stdio")
_mcs.stdio_client = lambda *a, **k: None
_mcs.StdioServerParameters = object
sys.modules.update({"mcp": _mcp, "mcp.client": _mc, "mcp.client.stdio": _mcs})

from jarvis.config import load_settings
from jarvis.memory.db import Database
from jarvis.memory.conversation import DialogueMemory
from jarvis.llm.factory import get_llm_backend
from jarvis.reply.engine import run_reply_engine
from jarvis.learning.author import LearningAuthor
from jarvis.learning.store import LearnedStore
from jarvis.learning.dataset import TrainingDataset
from jarvis.skills.manager import SkillManager

OK = []
FAIL = []


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("ok  " if cond else "FAIL") + " " + name)


tmp = tempfile.mkdtemp(prefix="jarvis_e2e_")
skills_dir = os.path.join(tmp, "skills")
learned_path = os.path.join(tmp, "learned.json")
pairs_path = os.path.join(tmp, "pairs.jsonl")
db_path = os.path.join(tmp, "jarvis_test.db")

# Real settings, then override the learning/skill flags + model.
# Settings is frozen, so use dataclasses.replace.
cfg = load_settings()
cfg = replace(
    cfg,
    llm_chat_model="gemma2:2b",
    embedding_model="gemma2:2b",
    skill_system_enabled=True,
    learning_enabled=True,
    implicit_corrections_enabled=True,
    skills_dir=skills_dir,
)

# Real Database (sqlite; vss optional) in the temp dir, not ~/.jarvis.
db = Database(db_path)

dm = DialogueMemory()

# Pre-seed a prior reply so implicit correction has something to react to.
dm.add_message("user", "what is the capital of France?")
dm.add_message("assistant", "The capital of France is Berlin.")  # intentionally wrong

print("=== A. Real LLM generation via Jarvis OllamaBackend (gemma2:2b) ===")
# Build the real Ollama backend the engine itself uses, and generate one reply.
backend = get_llm_backend(cfg)
gen = backend.chat(
    cfg.llm_chat_model,
    [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}],
    timeout_sec=60.0,
)
reply = (gen or {}).get("message", {}).get("content", "") if isinstance(gen, dict) else ""
check("Ollama backend returns a non-empty reply", bool(reply) and len(reply.strip()) > 0)
print("   reply:", repr(reply[:120]))

print("=== B. Authoring with isolated temp state (no ~ pollution) ===")
store = LearnedStore(learned_path)
ds = TrainingDataset(pairs_path)
author = LearningAuthor(enabled=True, store=store, dataset=ds, skills_dir=skills_dir)

r1 = author.handle("remember that I drink coffee at 9am", query="remember that I drink coffee at 9am")
check("remember returns confirmation", "remember" in (r1 or "").lower())
check("learned.json persisted the fact", any("coffee" in i for i in store.instructions))

r2 = author.handle(
    "create a skill: Standup Notes that triggers on standup, daily update",
    query="create a skill: Standup Notes that triggers on standup, daily update",
)
import glob
written = glob.glob(os.path.join(skills_dir, "*.md"))
check("create skill writes a .md file", bool(written))

r3 = author.handle(
    "actually, the capital is Paris",
    query="what is the capital of France?",
    last_reply="The capital of France is Berlin.",
)
check("implicit correction returns confirmation", "better" in (r3 or "").lower())
check("implicit correction captured a pair", len(ds.all()) == 1)
check("captured pair rejected the wrong reply",
      ds.all()[0]["rejected"] == "The capital of France is Berlin.")
check("captured pair chosen is the correction", "Paris" in ds.all()[0]["chosen"])

print("=== C. Skill discovery + context injection (isolated) ===")
mgr = SkillManager(skills_dir=skills_dir, enabled=True)
mgr.discover()
ctx = mgr.build_skill_context("give me my standup notes")
check("discovered the created skill", "standupNotes" in mgr.skills)
check("skill context built for matching query", bool(ctx) and "Standup" in ctx)

print("=== D. Finetune script dry-run on the captured pairs ===")
# Write the captured pairs to a file the script can read.
with open(os.path.join(tmp, "training_pairs.jsonl"), "w", encoding="utf-8") as f:
    for row in ds.all():
        f.write(json.dumps(row) + "\n")
from importlib import util as _u
spec = _u.spec_from_file_location(
    "finetune_lora", os.path.join(os.path.dirname(__file__), "scripts", "finetune_lora.py")
)
ft = _u.module_from_spec(spec)
spec.loader.exec_module(ft)
meta = ft._prepare(os.path.join(tmp, "training_pairs.jsonl"), os.path.join(tmp, "finetune_out"))
check("finetune prep read the pair", meta["num_pairs"] == 1)
check("finetune wrote llama.cpp dataset", os.path.exists(os.path.join(tmp, "finetune_out", "llamacpp_train.jsonl")))

print("=== E. Full run_reply_engine turn (real planner + Ollama) ===")
# Slow: runs the whole planner + a real Ollama generation on CPU. Only runs
# when JARVIS_E2E_ENGINE=1 so the default suite stays fast (a couple seconds).
if os.environ.get("JARVIS_E2E_ENGINE") == "1":
    dm2 = DialogueMemory()
    engine_reply = run_reply_engine(db, cfg, None, "What is 2 + 2?", dm2, language="en")
    check("run_reply_engine returns a non-empty reply on Python 3.12",
          bool(engine_reply) and len(engine_reply.strip()) > 0)
    print("   reply:", repr(engine_reply[:160]))
else:
    print("  (skipped: set JARVIS_E2E_ENGINE=1 to run the full-engine turn)")

db.close()
shutil.rmtree(tmp, ignore_errors=True)

print()
print(f"RESULT: {len(OK)} ok, {len(FAIL)} fail")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL GREEN - the chat-driven learning + skills system works end to end on this machine.")
