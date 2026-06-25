#!/usr/bin/env python3
"""Build the fine-tuning dataset from raw deployment logs.

Implements the dataset-construction pipeline of Section 4 of the paper:

1. **Curation** -- drop sessions truncated by network errors or operator
   intervention, near-duplicate instructions within a session, and
   sessions with hardware faults unrelated to the grounding decision.
2. **Serialisation** -- each curated trajectory becomes one multi-turn
   chat transcript that mirrors the inference-time prompt structure
   exactly: system prompt, user turn (instruction + mode), alternating
   assistant turns and observation injections, final verified action turn.
3. **Token augmentation** -- every assistant turn is prefixed with a
   *level token* (high-level vs. primitive) and a *verdict token* set to
   the trajectory's eventual monitor-derived outcome.
4. **Verdict-aware balancing** -- failed/partial trajectories (collapsed
   to a single FAILED class) are re-weighted so roughly half of the
   training mini-batches contain a failed trajectory.

Input: JSONL, one trajectory per line, with the schema in
``docs/DATASET.md``. Output: HuggingFace-ready JSONL of chat transcripts
plus per-example sampling weights and an ID/OOD split manifest.

Usage:
    python training/build_dataset.py \
        --logs data/trajectories/raw_sessions.jsonl \
        --out data/trajectories/processed \
        --ood-fraction 0.2
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from stretch_llm.llm.grammar import HIGH_LEVEL_BEHAVIOURS
from stretch_llm.llm.parser import extract_name
from stretch_llm.llm.prompts import SYSTEM_PROMPT

LEVEL_TOKEN = {1: "<|level:primitive|>", 2: "<|level:high|>"}
VERDICT_TOKEN = {"success": "<|verdict:success|>", "failed": "<|verdict:failed|>"}
SPECIAL_TOKENS = list(LEVEL_TOKEN.values()) + list(VERDICT_TOKEN.values())


def infer_level(commands):
    """Level 2 when the turn invokes any high-level behaviour, else Level 1."""
    for c in commands:
        if extract_name(c) in HIGH_LEVEL_BEHAVIOURS:
            return 2
    return 1


def collapse_verdict(verdict):
    """Two-class formulation: PARTIAL collapses into FAILED."""
    return "success" if verdict == "success" else "failed"


def is_near_duplicate(instr, seen, threshold=0.9):
    """Cheap near-duplicate test on lowercased token Jaccard similarity."""
    tokens = set(instr.lower().split())
    for prev in seen:
        union = tokens | prev
        if union and len(tokens & prev) / len(union) >= threshold:
            return True
    seen.append(tokens)
    return False


def serialise(traj):
    """Turn one curated trajectory into a token-augmented chat transcript."""
    verdict = collapse_verdict(traj["verdict"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (f"Current robot mode: {traj['mode']}\n\n"
                     f"User request: {traj['instruction']}")},
    ]
    for turn in traj["turns"]:
        if turn["role"] == "assistant":
            level = turn.get("level") or infer_level(turn["commands"])
            prefix = f"{VERDICT_TOKEN[verdict]} {LEVEL_TOKEN[level]} "
            messages.append({"role": "assistant",
                             "content": prefix + ", ".join(turn["commands"])})
        elif turn["role"] == "observation":
            messages.append({"role": "user",
                             "content": "CURRENT OBSERVATION:\n" + turn["text"]})
        elif turn["role"] == "answer":
            level = turn.get("level", 1)
            prefix = f"{VERDICT_TOKEN[verdict]} {LEVEL_TOKEN[level]} "
            messages.append({"role": "assistant",
                             "content": prefix + "ANSWER: " + turn["text"]})
    return {"messages": messages, "verdict": verdict,
            "trajectory_id": traj["trajectory_id"]}


def curate(trajectories):
    kept, seen_per_session, dropped = [], {}, Counter()
    for traj in trajectories:
        if traj.get("truncated") or traj.get("operator_intervention"):
            dropped["truncated_or_intervened"] += 1
            continue
        if traj.get("hardware_fault"):
            dropped["hardware_fault"] += 1
            continue
        seen = seen_per_session.setdefault(traj["session_id"], [])
        if is_near_duplicate(traj["instruction"], seen):
            dropped["near_duplicate"] += 1
            continue
        kept.append(traj)
    return kept, dropped


def split_id_ood(examples, ood_fraction, seed=13):
    """Deterministic instruction-level split.

    OOD membership is decided by a hash of the instruction text, so
    paraphrases of held-out instructions never leak into training.
    """
    train, ood = [], []
    for ex in examples:
        instr = ex["messages"][1]["content"]
        h = int(hashlib.sha256((str(seed) + instr).encode()).hexdigest(), 16)
        (ood if (h % 1000) / 1000.0 < ood_fraction else train).append(ex)
    return train, ood


def sampling_weights(train):
    """Verdict-balanced weights: ~50% of mini-batches see a failed trajectory."""
    n_succ = sum(1 for ex in train if ex["verdict"] == "success")
    n_fail = len(train) - n_succ
    for ex in train:
        ex["weight"] = (0.5 / n_succ) if ex["verdict"] == "success" \
            else (0.5 / max(n_fail, 1))
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ood-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    raw = [json.loads(line) for line in open(args.logs)]
    curated, dropped = curate(raw)
    examples = [serialise(t) for t in curated]
    train, ood = split_id_ood(examples, args.ood_fraction, args.seed)
    train = sampling_weights(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("test_ood", ood)):
        with open(out / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
    with open(out / "stats.json", "w") as f:
        json.dump({
            "raw": len(raw), "curated": len(curated), "dropped": dict(dropped),
            "train": len(train), "test_ood": len(ood),
            "train_failed": sum(1 for e in train if e["verdict"] == "failed"),
            "special_tokens": SPECIAL_TOKENS,
        }, f, indent=2)
    print(f"Curated {len(curated)}/{len(raw)} trajectories "
          f"-> {len(train)} train / {len(ood)} OOD test. Dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()
