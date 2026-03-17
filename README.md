# Legal Contract Review Agent

Fine tuned Mistral 7B on CUAD for automated legal contract review across 41 clause categories.



## What it does
Uploads a contract (PDF or TXT), reviews it across 41 clause types from the CUAD benchmark, 
flags legal risks, and returns a structured report. 

Built on a Mistral 7B model fine-tuned 
with QLoRA on 18,819 expert annotated contract passages.


## Stack
- **Model:** Mistral 7B Instruct v0.3
- **Fine-tuning:** QLoRA (4-bit quantization + LoRA adapters, r=16)
- **Data:** CUAD - 510 commercial contracts, 41 clause categories, 13,000+ expert annotations
- **Compute:** Modal A10G
- **Serving:** Modal web endpoint + Streamlit UI

## Project structure
```
data/prepare_cuad.py    - CUAD data pipeline
train/train.py          - QLoRA fine-tuning on Modal
train/eval.py           - F1 evaluation vs baseline
agent/agent.py          - ContractReviewAgent (sliding window + 41 clause types)
agent/risk_rules.py     - Deterministic risk flagging
serve/serve.py          - Modal web endpoint
serve/app.py            - Streamlit UI

```


## Training Results
- **Train loss:** 0.2422 (final)
- **Loss curve:** 0.59 -> 0.07 -> 0.04 over 1176 steps
- **Duration:** 3h 45min on Modal A10G
- **Model size:** 52MB LoRA adapter on Mistral 7B base

## Results

| Model | Macro F1 | Eval method |
|-------|----------|-------------|
| Base Mistral 7B Instruct | **0.0752** | ContractReviewAgent |
| Fine-tuned on CUAD (QLoRA, 512 ctx) | 0.0429 | ContractReviewAgent |

**Deployed model:** Base Mistral 7B Instruct.
**Finding:** Context truncation at 512 tokens caused systematic degradation.




## Run it yourself
```bash
# Install dependencies
pip install -r requirements.txt

# Fine-tune (requires Modal + HuggingFace token)
modal run --detach train/train.py

# Evaluate
modal run train/eval.py

# Deploy
modal deploy serve/serve.py
streamlit run serve/app.py
```

