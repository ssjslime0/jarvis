#!/usr/bin/env python3
"""
Local fine-tune preparation and orchestration for Jarvis.

This script turns the correction pairs captured by the chat-driven learning
subsystem (``~/.jarvis/training_pairs.jsonl``) into a model you can load back
into Ollama. It is deliberately dependency-light and offline-first:

- The **data preparation** step (reading the pairs, normalising them, and
  writing trainer-ready datasets) uses only the Python standard library.
- The **training** step delegates the heavy maths to a trainer you choose:
  * ``llama.cpp`` (``finetune`` binary) for GGUF models on CPU, or
  * HuggingFace ``transformers`` + ``peft`` when those packages are installed.

On a 6 GB RAM, CPU-only machine, ``llama.cpp`` is the realistic path for a
``gemma2:2b`` GGUF. This script does NOT bundle or download a trainer; it
prepares the data and prints the exact command to run, and will shell out to
the trainer only when you pass ``--run`` and the binary/backend is present.

Fail-open: if no trainer is available, the script still prepares the dataset
and exits with a clear message rather than pretending to train.

Example
-------
    # Prepare the datasets (no training yet):
    python scripts/finetune_lora.py --base-model ~/.jarvis/models/gemma2-2b.gguf

    # Actually train with llama.cpp if the finetune binary is on PATH:
    python scripts/finetune_lora.py --base-model gemma2-2b.gguf --run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PAIRS = os.path.join(
    os.path.expanduser("~"), ".jarvis", "training_pairs.jsonl"
)


def _read_pairs(path: str) -> List[Dict[str, str]]:
    """Read our training_pairs.jsonl into a list of dicts. Fail-open: returns
    [] on missing/unreadable input."""
    pairs: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return pairs
    # utf-8-sig transparently strips a leading BOM if present.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            prompt = (row.get("prompt") or "").strip()
            chosen = (row.get("chosen") or "").strip()
            if not prompt or not chosen:
                continue
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": (row.get("rejected") or "").strip(),
                }
            )
    return pairs


def _write_llamacpp_dataset(pairs: List[Dict[str, str]], out_path: Path) -> int:
    """Write a llama.cpp ``finetune``-compatible JSONL.

    llama.cpp finetune expects one JSON object per line, each with a
    ``text`` field (the full training sample). We build a supervised
    sample from ``prompt -> chosen``.
    """
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            sample = f"### Instruction:\n{p['prompt']}\n\n### Response:\n{p['chosen']}\n"
            f.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_sft_jsonl(pairs: List[Dict[str, str]], out_path: Path) -> int:
    """Write a plain instruction/response JSONL (HF SFT / generic)."""
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {"instruction": p["prompt"], "output": p["chosen"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def _write_dpo_jsonl(pairs: List[Dict[str, str]], out_path: Path) -> int:
    """Write a preference JSONL (HF DPO) when a rejected answer exists."""
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            if not p["rejected"]:
                continue
            f.write(
                json.dumps(
                    {
                        "prompt": p["prompt"],
                        "chosen": p["chosen"],
                        "rejected": p["rejected"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def _find_llamacpp_binary() -> Optional[str]:
    """Locate a llama.cpp finetune binary on PATH (best effort)."""
    for name in ("llama-finetune", "llama_finetune", "finetune", "llama.cpp-finetune"):
        found = shutil.which(name)
        if found:
            return found
    env = os.environ.get("LLAMACPP_FINETUNE_BIN")
    if env and os.path.exists(env):
        return env
    return None


def _prepare(pairs_path: str, out_dir: str) -> Dict[str, int]:
    pairs = _read_pairs(pairs_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sft = _write_sft_jsonl(pairs, out / "sft.jsonl")
    dpo = _write_dpo_jsonl(pairs, out / "dpo.jsonl")
    llama = _write_llamacpp_dataset(pairs, out / "llamacpp_train.jsonl")
    meta = {
        "source_pairs": pairs_path,
        "num_pairs": len(pairs),
        "sft_samples": sft,
        "dpo_samples": dpo,
        "llamacpp_samples": llama,
    }
    with open(out / "prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _run_llamacpp(base_model: str, out_dir: str, epochs: int, lr: float) -> int:
    bin_path = _find_llamacpp_binary()
    if not bin_path:
        print(
            "No llama.cpp finetune binary found on PATH.\n"
            "Install llama.cpp (build the `finetune` target) or set the\n"
            "LLAMACPP_FINETUNE_BIN environment variable to its path.",
            file=sys.stderr,
        )
        return 2
    train_file = os.path.join(out_dir, "llamacpp_train.jsonl")
    out_file = os.path.join(out_dir, "jarvis_lora.gguf")
    cmd = [
        bin_path,
        "--model", base_model,
        "--train-data", f"file://{train_file}",
        "--outfile", out_file,
        "--epochs", str(epochs),
        "--learning-rate", str(lr),
        "--no-use-checkpointing",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def _run_hf(base_model: str, out_dir: str, epochs: int, lr: float) -> int:
    """Fine-tune with HuggingFace PEFT/transformers (lazy import).

    All ML imports happen inside this function so the data-preparation path
    (the common case on a CPU-only box without torch installed) stays
    dependency-free.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        print(
            f"HuggingFace backend unavailable: {exc}\n"
            "Install torch, transformers, datasets and peft to use --backend hf.",
            file=sys.stderr,
        )
        return 2

    sft_path = os.path.join(out_dir, "sft.jsonl")
    records = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {"text": f"### Instruction:\n{obj['instruction']}\n\n### Response:\n{obj['output']}\n"}
            )
    if not records:
        print("No SFT samples to train on.", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, device_map="cpu"
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def _tok(ex):
        out = tokenizer(
            ex["text"], truncation=True, max_length=512, padding="max_length"
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_list(records).map(_tok)
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "hf_adapter"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "hf_adapter"))
    print(f"HuggingFace LoRA adapter saved to {os.path.join(out_dir, 'hf_adapter')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run a local LoRA fine-tune from Jarvis learning pairs."
    )
    parser.add_argument(
        "--pairs",
        default=DEFAULT_PAIRS,
        help="Path to training_pairs.jsonl (default: ~/.jarvis/training_pairs.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.expanduser("~"), ".jarvis", "finetune"),
        help="Directory for prepared datasets and output adapter.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path (or HuggingFace id) of the base model to fine-tune "
        "(e.g. a gemma2:2b GGUF file for llama.cpp).",
    )
    parser.add_argument(
        "--backend",
        choices=["llamacpp", "hf"],
        default="llamacpp",
        help="Trainer backend. llamacpp is the CPU/GGUF path; hf needs torch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs (default 3)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default 1e-4)."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually invoke the trainer. Without this, only data is prepared "
        "and the command is printed.",
    )
    args = parser.parse_args()

    meta = _prepare(args.pairs, args.out_dir)
    print(
        f"Prepared {meta['num_pairs']} pair(s) -> {args.out_dir}\n"
        f"  SFT samples:  {meta['sft_samples']}  (sft.jsonl)\n"
        f"  DPO samples:  {meta['dpo_samples']}  (dpo.jsonl)\n"
        f"  llama.cpp:    {meta['llamacpp_samples']}  (llamacpp_train.jsonl)"
    )

    if not args.run:
        if meta["num_pairs"] == 0:
            print(
                "\nNo training pairs yet. Teach Jarvis in chat (e.g. 'no, better: ...') "
                "and come back once pairs have accumulated."
            )
        else:
            print(
                "\nDry run: data prepared. Re-run with --run to train, or copy the "
                "command below for your backend."
            )
            if args.backend == "llamacpp":
                bin_hint = _find_llamacpp_binary() or "<llama.cpp finetune binary>"
                print(
                    f"  {bin_hint} --model {args.base_model} "
                    f"--train-data file://{os.path.join(args.out_dir, 'llamacpp_train.jsonl')} "
                    f"--outfile {os.path.join(args.out_dir, 'jarvis_lora.gguf')} "
                    f"--epochs {args.epochs} --learning-rate {args.lr}"
                )
        return 0

    if args.backend == "llamacpp":
        return _run_llamacpp(args.base_model, args.out_dir, args.epochs, args.lr)
    return _run_hf(args.base_model, args.out_dir, args.epochs, args.lr)


if __name__ == "__main__":
    raise SystemExit(main())
