# Skill System

Skills let a small local model gain capability without weight changes. A skill
is a Markdown file under `~/.jarvis/skills/` (or `skills_dir`) with a short
frontmatter header and a prompt body. When a query matches a skill's triggers,
the skill's prompt fragment is injected into the system context, so the model
behaves as if it had that skill for the turn.

## Skill file format

```markdown
name: Morning Briefing
description: Gives a concise start-of-day summary
triggers: morning, good morning, standup, daily briefing
---
When the user starts their day, lead with the current local time and one
practical suggestion. Keep it to three short sentences.
```

- `name` (required, human readable): also becomes the camelCase registry key
  (`Morning Briefing` -> `morningBriefing`).
- `description` (optional): what the skill does.
- `triggers` (optional, comma-separated): lowercase keywords matched against the
  query. A skill matches if any trigger is a substring of the query.
- `tool` (optional): name of a registered plugin tool the skill pairs with. The
  binding is lazy and only consulted when the referenced plugin is registered;
  absence never breaks the skill.
- The body after the `---` separator is the prompt fragment injected verbatim.

Files starting with `_` and non-`.md` files are ignored.

## Discovery

`SkillManager.discover()` reads every `*.md` skill file. Failures are logged
via `debug_log` and do not stop discovery. The daemon calls `discover()` at
startup when `skill_system_enabled` is true.

## Matching and injection

`SkillManager.get_relevant_skills(query)` returns skills whose triggers match.
`build_skill_context(query)` returns a `<Skills>`-style block of matching
prompts, or `None` when nothing matches or the system is disabled. The reply
engine appends this block to the system prompt only for the current query, so
the context stays small for a small model.

## Authoring

`LearningAuthor` (see `learning.spec.md`) can create skills at runtime via the
"create a skill: <name> that triggers on <keywords>" utterance; it writes a
skill file through `SkillManager.add_skill`.

## Constraints

- Skills are prompt context only; they do not change model weights.
- All processing is local; no network egress.
- Fail-open: discovery or injection errors are logged and degrade to no skill
  context, never breaking a reply.
- User-authored skill files run with the same privileges as the daemon; there
  is no `eval`/`exec` of skill text.
