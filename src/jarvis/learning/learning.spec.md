# Chat-Driven Learning

The learning subsystem turns live chat into a training surface. The user teaches
Jarvis through ordinary utterances; these are intercepted, handled, and
persisted locally, so capability accumulates over time without bigger hardware.

## Three explicit triggers (Phase 1)

| Utterance | Intent | Effect |
|-----------|--------|--------|
| `remember that <fact>` | REMEMBER | Append a learned instruction to `~/.jarvis/learned.json` |
| `create a skill: <name> that triggers on <kw1>, <kw2>` | CREATE_SKILL | Write a skill Markdown file via `SkillManager.add_skill` |
| `no, better: <preferred reply>` | CORRECT | Append a `(prompt, chosen, rejected)` pair to `~/.jarvis/training_pairs.jsonl` |

Regex pre-checks these triggers so they never depend on the model. An optional
injected `classifier` callable (e.g. the small router model) may upgrade a
NORMAL turn to a teaching intent, but it can only add signals; regex wins for
the three explicit triggers.

## Storage

- `LearnedStore` persists learned instructions as a flat JSON list
  (`learned.json`). `build_context()` renders them as a `<Learned>` block for
  injection. Dedupes identical instructions on add.
- `TrainingDataset` appends preference pairs as JSONL (`training_pairs.jsonl`),
  export-ready for a later LoRA/QLoRA fine-tune of the local model.

All persistence is fail-open: read/write errors are logged and degrade to an
empty/in-memory state rather than breaking a reply.

## Injection

`run_reply_engine` runs `LearningAuthor.handle` as Step 0. If it returns a
confirmation, the engine returns it directly (no full reply generation). The
learned-instructions block is also injected into the system prompt when
`learning_enabled` is true, so future turns honour what was taught.

## Roadmap

- Phase 2 (implemented): implicit correction capture. When there is a prior
  reply and the user's message reads as feedback on it (e.g. "actually, ...",
  "you're wrong, ...", "I meant ...", "instead, ...", "shorter next time")
  without being a fresh question, it is captured as a correction pair (chosen =
  the user's phrasing with the lead-in stripped, rejected = the prior reply).
  Gated behind `implicit_corrections_enabled` (default true) and only fires
  with a non-empty prior reply and a non-question utterance, to keep false
  positives low on a small model.
- Phase 3: a local fine-tune script (`scripts/finetune_lora.py`) that consumes
  `training_pairs.jsonl` to produce a custom `gemma2:2b` variant reloaded into
  Ollama.

## Constraints

- Local only; no network egress.
- Fail-open at every stage.
- British English, no em dashes, in any user-facing copy.
- No `eval`/`exec` of chat text; skills/learned data are plain files.
