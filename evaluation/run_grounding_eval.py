#!/usr/bin/env python3
"""Run the grounding evaluation on a held-out OOD split.

Queries an Ollama-served model on every test instruction, parses the
response with the deployment-time parser, and reports the metric suite
of Section 7 (EM, SM, TFC, MVR, hallucination rate, level accuracy).

Usage:
    python evaluation/run_grounding_eval.py \
        --data data/trajectories/processed/test_ood.jsonl \
        --model llama3.1:latest --url http://127.0.0.1:11435 \
        --out results/e2_ft_real.json
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import requests

from evaluation.metrics import Example, evaluate
from stretch_llm.llm.parser import extract_commands

LEVEL_RE = re.compile(r"<\|level:(primitive|high)\|>")


def first_assistant_turn(messages):
    for m in messages:
        if m["role"] == "assistant":
            return m["content"]
    return ""


def query(url, model, system, user):
    r = requests.post(f"{url}/api/generate", json={
        "model": model, "prompt": f"{system}\n\nUser: {user}\nAssistant:\n",
        "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def level_of(text):
    m = LEVEL_RE.search(text)
    if not m:
        return None
    return 1 if m.group(1) == "primitive" else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    examples, latencies = [], []
    for line in open(args.data):
        ex = json.loads(line)
        system = ex["messages"][0]["content"]
        user = ex["messages"][1]["content"]
        reference_turn = first_assistant_turn(ex["messages"])

        t0 = time.time()
        raw = query(args.url, args.model, system, user)
        latencies.append(time.time() - t0)

        examples.append(Example(
            predicted=extract_commands(LEVEL_RE.sub("", raw)),
            reference=extract_commands(LEVEL_RE.sub("", reference_turn)),
            requires_tool_first="safe" in user.lower()
                                or "obstacle" in user.lower(),
            predicted_level=level_of(raw),
            reference_level=level_of(reference_turn),
        ))

    report = evaluate(examples).as_dict()
    report["median_latency_ms"] = round(
        statistics.median(latencies) * 1000, 1)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
