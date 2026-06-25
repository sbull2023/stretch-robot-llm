#!/usr/bin/env python3
"""Level-aware verdict-conditioned LoRA fine-tuning.

Trains either the Tier-1 primary model (Llama-3.1-8B-Instruct) or the
Tier-2 monitor (Gemma-3-4B) on the trajectory dataset produced by
``build_dataset.py``. Hyperparameters default to the ones reported in
Section 4.4 of the paper:

* LoRA rank 16, alpha 32, on q_proj / k_proj / v_proj / o_proj,
  base weights frozen
* learning rate 1e-4, effective batch size 32 via gradient accumulation
* 3 epochs, AdamW, cosine schedule, 500-step warmup
* maximum sequence length 8192 tokens (longest trajectory fits)

Two details matter beyond the hyperparameters. First, the level and
verdict tokens are added to the tokenizer as special tokens and the
embedding matrix is resized, so the conditioning variables occupy
dedicated vocabulary slots rather than sub-word pieces. Second, the
trainer consumes the per-example ``weight`` column from the dataset
builder, which realises verdict-balanced sampling without duplication of
failed trajectories on disk.

Run on AI-Panther (4x A100 80GB; ~18 h per model):
    sbatch scripts/hpc/train_lora.sbatch
or directly:
    python training/finetune_lora.py --model meta-llama/Llama-3.1-8B-Instruct \
        --data data/trajectories/processed --out runs/llama31-8b-traj-lora
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
from torch.utils.data import WeightedRandomSampler

SPECIAL_TOKENS = ["<|level:primitive|>", "<|level:high|>",
                  "<|verdict:success|>", "<|verdict:failed|>"]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", required=True,
                    help="Directory with train.jsonl from build_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)  # 2 x 16 = 32 effective
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=500)
    return ap


class WeightedTrainer(Trainer):
    """Trainer with a verdict-balanced WeightedRandomSampler."""

    def __init__(self, sample_weights=None, **kwargs):
        super().__init__(**kwargs)
        self._sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        if self._sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        return WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=len(self._sample_weights),
            replacement=True,
        )


def main():
    args = build_argparser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw = load_dataset("json",
                       data_files=str(Path(args.data) / "train.jsonl"))["train"]
    weights = raw["weight"] if "weight" in raw.column_names else None

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=args.max_len)

    dataset = raw.map(tokenize, remove_columns=raw.column_names)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = WeightedTrainer(
        sample_weights=weights,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(Path(args.out) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
