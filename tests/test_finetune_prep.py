"""Tests for the fine-tune data-preparation script (scripts/finetune_lora.py).

Only the stdlib data-prep path is exercised (no torch/llama.cpp required).
Verifies pair reading, normalisation, and dataset writing for each backend
format, plus dry-run behaviour when there are no pairs.
"""

import sys
import os
import json
import tempfile
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# scripts/ is not a package; load the module directly.
_spec = importlib.util.spec_from_file_location(
    "finetune_lora", os.path.join(os.path.dirname(__file__), "..", "scripts", "finetune_lora.py")
)
finetune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(finetune)


def _write_pairs(tmp, rows):
    path = os.path.join(tmp, "pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_read_pairs_skips_malformed_and_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_pairs(
            tmp,
            [
                {"prompt": "what is 2+2?", "chosen": "4"},
                {"prompt": "", "chosen": "x"},  # missing prompt
                "not json",
                {"chosen": "only chosen"},  # missing prompt
            ],
        )
        pairs = finetune._read_pairs(path)
        assert len(pairs) == 1
        assert pairs[0]["prompt"] == "what is 2+2?"
        assert pairs[0]["chosen"] == "4"


def test_read_pairs_missing_file_returns_empty():
    assert finetune._read_pairs(os.path.join("nonexistent", "pairs.jsonl")) == []


def test_prepare_writes_all_formats():
    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = _write_pairs(
            tmp,
            [
                {"prompt": "greet me", "chosen": "Hello!", "rejected": "Yo."},
                {"prompt": "time?", "chosen": "It is 3pm.", "rejected": ""},
            ],
        )
        out = os.path.join(tmp, "out")
        meta = finetune._prepare(pairs_path, out)
        assert meta["num_pairs"] == 2
        assert meta["sft_samples"] == 2
        assert meta["dpo_samples"] == 1  # only one has a rejected answer
        assert meta["llamacpp_samples"] == 2
        for name in ("sft.jsonl", "dpo.jsonl", "llamacpp_train.jsonl", "prepare_manifest.json"):
            assert os.path.exists(os.path.join(out, name)), name


def test_llamacpp_dataset_format():
    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = _write_pairs(
            tmp, [{"prompt": "say hi", "chosen": "Hi there."}]
        )
        out = os.path.join(tmp, "out")
        finetune._prepare(pairs_path, out)
        with open(os.path.join(out, "llamacpp_train.jsonl"), encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert "text" in row
        assert "### Instruction:" in row["text"]
        assert "say hi" in row["text"]
        assert "Hi there." in row["text"]


def test_dpo_dataset_format():
    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = _write_pairs(
            tmp, [{"prompt": "x", "chosen": "good", "rejected": "bad"}]
        )
        out = os.path.join(tmp, "out")
        finetune._prepare(pairs_path, out)
        with open(os.path.join(out, "dpo.jsonl"), encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row["prompt"] == "x"
        assert row["chosen"] == "good"
        assert row["rejected"] == "bad"


def test_find_llamacpp_binary_absent_returns_none():
    # When the binary is genuinely absent and env unset, returns None.
    saved = os.environ.pop("LLAMACPP_FINETUNE_BIN", None)
    try:
        import shutil

        orig = shutil.which
        shutil.which = lambda *a, **k: None
        assert finetune._find_llamacpp_binary() is None
        shutil.which = orig
    finally:
        if saved:
            os.environ["LLAMACPP_FINETUNE_BIN"] = saved


def test_read_pairs_handles_utf8_bom():
    import codecs

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bom.jsonl")
        with codecs.open(path, "w", encoding="utf-8-sig") as f:
            f.write(json.dumps({"prompt": "hi", "chosen": "hello"}) + "\n")
        pairs = finetune._read_pairs(path)
        assert len(pairs) == 1
        assert pairs[0]["prompt"] == "hi"
