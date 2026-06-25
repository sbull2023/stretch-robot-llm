# Fine-Tuning Recipe

## Pipeline

```
raw_sessions.jsonl
   │  curation: drop truncated / intervened / hardware-fault /
   │            near-duplicate trajectories
   ▼
build_dataset.py ──► train.jsonl + test_ood.jsonl + stats.json
   │  serialisation into chat transcripts that mirror the
   │  inference prompt exactly; level + verdict token prefixes;
   │  verdict-balanced sample weights
   ▼
finetune_lora.py ──► LoRA adapter (per model)
   ▼
verdict_inference.py / run_grounding_eval.py
```

## Hyperparameters (Section 4.4 of the paper)

| Setting | Value |
|---|---|
| Method | LoRA, base weights frozen |
| Rank / alpha | 16 / 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 1e-4, cosine schedule, 500-step warmup |
| Effective batch size | 32 (gradient accumulation) |
| Epochs | 3 |
| Max sequence length | 8192 tokens |
| Compute | 4× A100 80 GB on AI-Panther, ~18 h per model |

## Special tokens

`<|level:primitive|>`, `<|level:high|>`, `<|verdict:success|>`,
`<|verdict:failed|>` are added to the tokenizer as dedicated vocabulary
entries before training; the embedding matrix is resized accordingly.

## Verdict-balanced sampling

Failed/partial trajectories (~17% of the curated set) are re-weighted so
roughly half of the mini-batches contain one. Without this, the failed
conditional is undertrained and the verdict-forcing diagnostic of E3
collapses.

## Inference

Deployment queries force the success prefix through the logit mask in
`training/verdict_inference.py`. The level token is left free: the model
chooses the abstraction level, and the choice is observable in the output.
