"""
Fine-tune Mistral 7B on CUAD using QLoRA.
"""

import modal
import json
import transformers

app = modal.App("legal-contract-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "datasets==2.20.0",
        "peft==0.12.0",
        "trl==0.9.6",
        "bitsandbytes==0.43.3",
        "accelerate==0.33.0",
        "scipy",
        "rich",
        "sentencepiece",
    )
)

volume = modal.Volume.from_name("cuad-sft-vol", create_if_missing=True)


# ── Checkpoint callback ───────────────────────────────────────────────────────

class VolumeCommitCallback(transformers.TrainerCallback):
    """Commits Modal volume after every checkpoint so progress is never lost."""
    def on_save(self, args, state, control, **kwargs):
        print(f"Committing volume at step {state.global_step}...")
        volume.commit()


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

def format_for_sft(example):
    context = example["context"]
    clause_question = example["clause_type"]
    is_impossible = example["is_impossible"]
    answers = example["answers"]

    if is_impossible or len(answers) == 0:
        assistant_response = "No relevant clause found in this passage."
    else:
        seen = set()
        unique_answers = []
        for a in answers:
            text = a["text"].strip()
            if text not in seen:
                seen.add(text)
                unique_answers.append(text)

        if len(unique_answers) == 1:
            assistant_response = f"Relevant clause found:\n\n{unique_answers[0]}"
        else:
            spans = "\n\n".join(
                f"{i+1}. {text}" for i, text in enumerate(unique_answers)
            )
            assistant_response = f"Relevant clauses found:\n\n{spans}"

    return {
        "messages": [
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
                "content": f"{clause_question}\n\nContract passage:\n{context}"
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]
    }

def build_hf_dataset(examples):
    from datasets import Dataset
    records = [format_for_sft(ex) for ex in examples]
    return Dataset.from_list(records)


# ── Training ──────────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    timeout=60 * 60 * 8,  # 8 hours — well above what we need
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print("Loading CUAD data...")
    data = load_cuad()
    examples = flatten_examples(data)

    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_dataset = build_hf_dataset(train_examples)
    val_dataset = build_hf_dataset(val_examples)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def formatting_func(batch):
        return [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
            for msgs in batch["messages"]
        ]

    training_args = SFTConfig(
        output_dir="/vol/cuad-mistral-v1",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        bf16=True,
        max_seq_length=512,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_func,
        callbacks=[VolumeCommitCallback()],
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model("/vol/cuad-mistral-final")
    tokenizer.save_pretrained("/vol/cuad-mistral-final")
    volume.commit()
    print("Done. Model saved to /vol/cuad-mistral-final")


@app.local_entrypoint()
def main():
    train.remote()