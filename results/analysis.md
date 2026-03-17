# Evaluation Analysis

## Results (via ContractReviewAgent.extract_clause_from_chunk)
| Model | Macro F1 |
|-------|----------|
| Base Mistral 7B Instruct | 0.0752 |
| Fine-tuned on CUAD (QLoRA, 512 ctx) | 0.0429 |

## Finding
Fine-tuning degraded performance by 43%. Root cause: context truncation.

## Why it failed
CUAD contracts average 10,000+ tokens. We trained with max_seq_length=512.
Most clause text appears beyond token 512, so the model saw truncated passages
with the answer cut off - and learned to output "no relevant clause found."

Evidence: clauses appearing early in contracts (Governing Law: 0.136 ft vs 
0.141 base, License Grant: 0.134 ft vs 0.153 base) show smallest degradation.
Clauses appearing late (Notice Period: 0.020 ft vs 0.115 base, Exclusivity: 
0.046 ft vs 0.129 base) show largest - exactly what truncation predicts.

## Eval method
Both models evaluated using ContractReviewAgent.extract_clause_from_chunk()
- same inference code as production serving.

## What we deploy
Base Mistral 7B Instruct. Higher F1, no fine-tuning cost at inference.

## What would fix it
- max_seq_length=2048 on A100
- Learning rate: 5e-5 instead of 2e-4  
- 2-3 epochs
- Expected outcome: F1 > 0.10, beating baseline