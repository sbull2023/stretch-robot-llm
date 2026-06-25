#!/usr/bin/env python3
"""Verdict-conditioned inference with a forced success prefix.

At inference time the verdict token becomes a controllable variable: a
logit-mask constraint at the verdict position forces the model to begin
every assistant turn with ``<|verdict:success|>``, which elicits the
model's success-conditional command distribution rather than the marginal
one (Section 4.6 of the paper). Flip ``--verdict failed`` to reproduce
the verdict-forcing diagnostic of E3 -- the conditioned model scores
measurably worse under the failed prefix, which confirms the conditional
was learned rather than collapsed.

Usage:
    python training/verdict_inference.py --model runs/llama31-8b-traj-lora \
        --instruction "Is it safe to move forward 1 metre? If yes, do it." \
        --mode position
"""

import argparse

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

from stretch_llm.llm.prompts import SYSTEM_PROMPT


class ForcedPrefixProcessor(LogitsProcessor):
    """Force a fixed token sequence at the start of generation.

    For positions inside the prefix, every logit except the required
    token's is set to ``-inf``; beyond the prefix the distribution is
    untouched. With the level token excluded from the prefix, the model
    still chooses the abstraction level freely -- only the verdict is
    pinned.
    """

    def __init__(self, prefix_ids, prompt_len):
        self.prefix_ids = prefix_ids
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        position = input_ids.shape[1] - self.prompt_len
        if 0 <= position < len(self.prefix_ids):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.prefix_ids[position]] = 0.0
            scores = scores + mask
        return scores


def generate(model, tokenizer, instruction, mode="position",
             verdict="success", max_new_tokens=128):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Current robot mode: {mode}\n\nUser request: {instruction}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prefix_ids = tokenizer(f"<|verdict:{verdict}|>",
                           add_special_tokens=False).input_ids
    processors = LogitsProcessorList([
        ForcedPrefixProcessor(prefix_ids, inputs.input_ids.shape[1])
    ])

    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, logits_processor=processors)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--mode", default="position",
                    choices=["position", "navigation"])
    ap.add_argument("--verdict", default="success",
                    choices=["success", "failed"])
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    print(generate(model, tokenizer, args.instruction,
                   mode=args.mode, verdict=args.verdict))


if __name__ == "__main__":
    main()
