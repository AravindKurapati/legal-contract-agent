# Evaluation Analysis

## Results
| Model | Macro F1 |
|-------|----------|
| Base Mistral 7B Instruct | 0.0725 |
| Fine-tuned on CUAD (QLoRA) | 0.0430 |

## Finding
Fine-tuning degraded performance by 40.8%. Root cause: context truncation.

## Why it failed
CUAD contracts average 10,000+ tokens. We trained with `max_seq_length=512`.
Most clause text appears beyond token 512, so the model saw truncated passages
with the answer cut off — and learned to output "no relevant clause found."

Evidence: clauses that appear early in contracts (Non-Compete, Anti-Assignment)
show the smallest degradation. Clauses that appear late (Governing Law,
Expiration Date, No-Solicit) show the largest — exactly what truncation predicts.

## What we deploy
Base Mistral 7B Instruct. Higher F1, no fine-tuning cost at inference.

## What would fix it
- `max_seq_length=2048` — requires A100 (~$1.50/hr vs $0.76 for A10G)
- Lower learning rate: 5e-5 instead of 2e-4
- 2-3 epochs instead of 1
- Expected outcome: F1 > 0.10, beating baseline by ~38%