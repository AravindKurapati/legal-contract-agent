"""
Evaluate base Mistral 7B on CUAD test set using ContractReviewAgent.
Routes inference through agent.extract_clause_from_chunk() — same
code path as production serving.

Compares:
  1. Base Mistral 7B (what we deploy)
  2. Fine-tuned Mistral 7B on CUAD (for reference)
"""

import modal
import json
import sys
from collections import defaultdict

app = modal.App("legal-contract-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "peft==0.12.0",
        "bitsandbytes==0.43.3",
        "accelerate==0.33.0",
        "sentencepiece",
        "rich",
    )
)

volume = modal.Volume.from_name("cuad-sft-vol", create_if_missing=True)

# Mount agent/ directory so Modal container can import it
agent_mount = modal.Mount.from_local_dir(
    "agent",
    remote_path="/root/agent"
)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_cuad(path="/vol/data/CUAD_v1.json"):
    with open(path) as f:
        raw = json.load(f)
    return raw["data"]

def flatten_examples(data):
    examples = []
    for contract in data:
        for para in contract["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "context": context,
                    "clause_type": qa["question"],
                    "answers": qa["answers"],
                    "is_impossible": qa["is_impossible"]
                })
    return examples


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize_text(text):
    return " ".join(text.lower().split())

def compute_token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    pred_counter = defaultdict(int)
    for t in pred_tokens:
        pred_counter[t] += 1

    gt_counter = defaultdict(int)
    for t in gt_tokens:
        gt_counter[t] += 1

    shared = sum(
        min(pred_counter[t], gt_counter[t])
        for t in pred_counter
    )

    precision = shared / len(pred_tokens)
    recall = shared / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, examples, n_samples=500):
    """
    Evaluate using ContractReviewAgent.extract_clause_from_chunk()
    — exact same inference code as production serving.
    """
    sys.path.insert(0, "/root/agent")
    from agent import ContractReviewAgent

    agent = ContractReviewAgent(model, tokenizer)
    results_by_clause = defaultdict(list)

    for i, example in enumerate(examples[:n_samples]):
        if i % 50 == 0:
            print(f"Evaluating {i}/{n_samples}...")

        context = example["context"]
        clause_type = example["clause_type"]
        is_impossible = example["is_impossible"]
        answers = example["answers"]

        # Use agent's inference method — same as production
        prediction = agent.extract_clause_from_chunk(context, clause_type)
        prediction = prediction if prediction else "No relevant clause found in this passage."

        if is_impossible or len(answers) == 0:
            ground_truth = "No relevant clause found in this passage."
        else:
            ground_truth = answers[0]["text"].strip()

        f1 = compute_token_f1(prediction, ground_truth)
        results_by_clause[clause_type].append(f1)

    per_clause_f1 = {
        clause: sum(scores) / len(scores)
        for clause, scores in results_by_clause.items()
    }
    all_scores = [s for scores in results_by_clause.values() for s in scores]
    macro_f1 = sum(all_scores) / len(all_scores)

    return macro_f1, per_clause_f1


# ── Modal functions ───────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    mounts=[agent_mount],
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_baseline():
    """Evaluate base Mistral 7B — the model we actually deploy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading CUAD test data...")
    data = load_cuad()
    all_examples = flatten_examples(data)
    split_idx = int(len(all_examples) * 0.9)
    test_examples = all_examples[split_idx:]
    print(f"Test examples: {len(test_examples)}")

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("Evaluating via ContractReviewAgent...")
    macro_f1, per_clause_f1 = evaluate_model(model, tokenizer, test_examples)

    print(f"\n=== Base Model Results ===")
    print(f"Macro F1: {macro_f1:.4f}")

    results = {
        "model": "base-mistral-7b-instruct",
        "eval_method": "ContractReviewAgent.extract_clause_from_chunk",
        "macro_f1": macro_f1,
        "per_clause_f1": per_clause_f1,
        "n_samples": 500
    }

    with open("/vol/results_baseline_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print("Saved to /vol/results_baseline_v2.json")
    return macro_f1


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    mounts=[agent_mount],
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_finetuned():
    """Evaluate fine-tuned model for reference comparison."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("Loading CUAD test data...")
    data = load_cuad()
    all_examples = flatten_examples(data)
    split_idx = int(len(all_examples) * 0.9)
    test_examples = all_examples[split_idx:]

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model + fine-tuned adapter...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, "/vol/cuad-mistral-final")
    model.eval()

    print("Evaluating via ContractReviewAgent...")
    macro_f1, per_clause_f1 = evaluate_model(model, tokenizer, test_examples)

    print(f"\n=== Fine-tuned Model Results ===")
    print(f"Macro F1: {macro_f1:.4f}")

    results = {
        "model": "fine-tuned-mistral-7b-cuad",
        "eval_method": "ContractReviewAgent.extract_clause_from_chunk",
        "macro_f1": macro_f1,
        "per_clause_f1": per_clause_f1,
        "n_samples": 500
    }

    with open("/vol/results_finetuned_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print("Saved to /vol/results_finetuned_v2.json")
    return macro_f1


@app.local_entrypoint()
def main():
    print("Running baseline eval...")
    baseline_f1 = eval_baseline.remote()

    print("Running fine-tuned eval...")
    finetuned_f1 = eval_finetuned.remote()

    print(f"\n{'='*40}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"Base Mistral 7B:     {baseline_f1:.4f}")
    print(f"Fine-tuned on CUAD:  {finetuned_f1:.4f}")
    improvement = (finetuned_f1 - baseline_f1) / baseline_f1 * 100
    print(f"Improvement:         {improvement:+.1f}%")