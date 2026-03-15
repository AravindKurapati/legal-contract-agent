# Training Spec: CUAD SFT Run v1

## Hypothesis
Fine-tuning Mistral 7B Instruct on CUAD with QLoRA will produce a model 
that extracts legal clauses more accurately than base Mistral 7B with 
just a system prompt.

## Data
- 20,910 examples across 41 clause types from 510 contracts
- 18,819 train / 2,091 val (90/10 split)
- 2.1x more negative examples than positive
- Format: chat messages with system/user/assistant roles

## Model
- Base: mistralai/Mistral-7B-Instruct-v0.3
- Method: QLoRA (4-bit + LoRA adapters)
- LoRA rank: r=16, alpha=32
- Target modules: q_proj, v_proj, k_proj, o_proj

## Training config
- Epochs: 3
- Effective batch size: 16 (2 per device × 8 gradient accumulation)
- Learning rate: 2e-4 with cosine decay
- Max seq length: 1024
- Expected runtime: ~2-3 hours on A10G
- Expected cost: ~$2

## Expected outcome
- Training loss should decrease from ~2.0 to ~0.5 over 3 epochs
- Val loss should follow closely (no big gap = no overfitting)
- Model should correctly say "No relevant clause found" for negatives
- Model should extract correct spans for positives

## What we'll do if it fails
- Loss doesn't decrease → LR too high, try 1e-4
- OOM error → reduce per_device_train_batch_size to 1
- Val loss diverges from train loss → overfitting, reduce epochs to 2
- Model always says "No relevant clause found" → 
  data imbalance issue, need to oversample positives

## Eval plan
- Run eval.py on CUAD test set after training
- Report F1, precision, recall per clause type
- Compare against base Mistral 7B (no fine-tuning) as baseline