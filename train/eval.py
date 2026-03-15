"""
Evaluate fine-tuned Mistral 7B on CUAD test set.

"""

import modal
import json
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


def load_cuad(path):
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


def normalize_text(text):
    """Lowercase and remove extra whitespace."""
    return " ".join(text.lower().split())

def compute_token_f1(prediction, ground_truth):

    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0  # both empty = correct "not present"
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0  # one empty, one not = wrong

    # Count shared tokens
    pred_counter = defaultdict(int)
    for t in pred_tokens:
        pred_counter[t] += 1

    gt_counter = defaultdict(int)
    for t in gt_tokens:
        gt_counter[t] += 1

    shared = 0
    for token in pred_counter:
        shared += min(pred_counter[token], gt_counter[token])

    precision = shared / len(pred_tokens)
    recall = shared / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def run_inference(model, tokenizer, context, clause_type):

    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal contract review assistant. "
                "Given a contract passage and a question about a specific clause type, "
                "extract the relevant clause text if present. "
                "If the clause is not present in the passage, say so clearly."
            )
        },
        {
            "role": "user",
            "content": f"{clause_type}\n\nContract passage:\n{context}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # True at inference time 
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,   # clause text is usually under 200 tokens
            temperature=0.1,      # low temperature = more deterministic output
            do_sample=False,      # greedy decoding for eval
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the input prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def evaluate_model(model, tokenizer, examples, n_samples=500):
    results_by_clause = defaultdict(list)
    
    for i, example in enumerate(examples[:n_samples]):
        if i % 50 == 0:
            print(f"Evaluating {i}/{n_samples}...")

        context = example["context"]
        clause_type = example["clause_type"]
        is_impossible = example["is_impossible"]
        answers = example["answers"]

        # Get model prediction
        prediction = run_inference(model, tokenizer, context, clause_type)

        # Get ground truth
        if is_impossible or len(answers) == 0:
            ground_truth = "No relevant clause found in this passage."
        else:
            ground_truth = answers[0]["text"].strip()

        # Compute F1
        f1 = compute_token_f1(prediction, ground_truth)
        results_by_clause[clause_type].append(f1)

    # Aggregate
    per_clause_f1 = {
        clause: sum(scores) / len(scores)
        for clause, scores in results_by_clause.items()
    }

    all_scores = [s for scores in results_by_clause.values() for s in scores]
    macro_f1 = sum(all_scores) / len(all_scores)

    return macro_f1, per_clause_f1


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_finetuned():
    """Evaluate our fine tuned model"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("Loading CUAD test data...")
    data = load_cuad("/vol/data/CUAD_v1.json")
    all_examples = flatten_examples(data)

    # Use last 10% as test set (same split as training)
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

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        "/vol/cuad-mistral-final"
    )
    model.eval()

    print("Evaluating fine-tuned model...")
    macro_f1, per_clause_f1 = evaluate_model(model, tokenizer, test_examples)

    print(f"\n=== Fine-tuned Model Results ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-clause F1 (top 10):")
    for clause, f1 in sorted(per_clause_f1.items(), key=lambda x: -x[1])[:10]:
        print(f"  {f1:.3f}  {clause[:60]}")

    # Save results
    results = {
        "model": "fine-tuned-mistral-7b-cuad",
        "macro_f1": macro_f1,
        "per_clause_f1": per_clause_f1,
        "n_samples": 500
    }

    with open("/vol/results_finetuned.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to /vol/results_finetuned.json")
    return macro_f1

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_baseline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading CUAD test data...")
    data = load_cuad("/vol/data/CUAD_v1.json")
    all_examples = flatten_examples(data)
    split_idx = int(len(all_examples) * 0.9)
    test_examples = all_examples[split_idx:]

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("Evaluating base model...")
    macro_f1, per_clause_f1 = evaluate_model(model, tokenizer, test_examples)

    print(f"\n=== Base Model Results ===")
    print(f"Macro F1: {macro_f1:.4f}")

    results = {
        "model": "base-mistral-7b-instruct",
        "macro_f1": macro_f1,
        "per_clause_f1": per_clause_f1,
        "n_samples": 500
    }

    with open("/vol/results_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to /vol/results_baseline.json")
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