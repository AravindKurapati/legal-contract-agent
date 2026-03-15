"""
Fine-tune Mistral 7B on CUAD using QLoRA.
"""

import modal
import json
import os


app = modal.App("legal-contract-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "datasets==2.20.0",
        "peft==0.12.0",         # LoRA implementation
        "trl==0.9.6",           # SFTTrainer
        "bitsandbytes==0.43.3", # 4-bit quantization
        "accelerate==0.33.0",
        "scipy",
        "rich",
        "sentencepiece",
    )
)

# Modal volume to persist model checkpoints between runs
volume = modal.Volume.from_name("cuad-sft-vol", create_if_missing=True)


def load_cuad(path="data/CUAD_v1.json"):
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
            spans = "\n\n".join(f"{i+1}. {text}" for i, text in enumerate(unique_answers))
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
    """Convert list of formatted examples into HuggingFace Dataset."""
    from datasets import Dataset

    # SFTTrainer expects a dataset with a 'text' column
    # We use the apply_chat_template step in training to convert messages -> text
    # just wrap the messages list
    records = [format_for_sft(ex) for ex in examples]
    return Dataset.from_list(records)


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume,
},
    timeout=60 * 60 * 4,  # 3 hour max
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print("Loading CUAD data...")
    # Note: CUAD_v1.json is uploaded separately via modal volume
    # For now we load from the mounted path
    data = load_cuad("/vol/data/CUAD_v1.json")
    examples = flatten_examples(data)

    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_dataset = build_hf_dataset(train_examples)
    val_dataset = build_hf_dataset(val_examples)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # quantize the quantization constants too
    )

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    tokenizer.padding_side = "right"           # pad on right for causal LM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # automatically place layers on GPU
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training
    # This does gradient checkpointing + casts layer norms to float32
    # Required step before applying LoRA to a quantized model
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ────────────────────────────────────────────────────────────
    # r=16: rank of the adapter matrices
    #   Higher r = more parameters = more expressive but slower
    #   16 is standard for 7B models on legal tasks
    #
    # lora_alpha=32: scaling factor = alpha/r = 2.0
    #   Controls how much the adapter influences the output
    #
    # target_modules: which weight matrices to adapt
    #   q_proj, v_proj = query and value projections in attention
    #   These are the standard targets for instruction tuning
    #
    # lora_dropout=0.05: small dropout for regularization
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
    # Expected output: ~1-2% of total parameters are trainable

    training_args = SFTConfig(
        output_dir="/vol/cuad-mistral-v1",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
       

        learning_rate=2e-4,
        lr_scheduler_type="cosine",  
        warmup_ratio=0.05,           # 5% of steps used for LR warmup

        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=200,              
        save_steps=200,              
        save_total_limit=2,          # keep only 2 most recent checkpoints

        logging_steps=50,
        bf16=True,                   # use bfloat16 for training
        max_seq_length=512,         # max token length per example
        # CUAD passages are long but 1024 covers most of them

        report_to="none",               # disable wandb/tensorboard for now
    )

    def formatting_func(batch):
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model("/vol/cuad-mistral-final")
    tokenizer.save_pretrained("/vol/cuad-mistral-final")
    print("Done. Model saved to /checkpoints/cuad-mistral-final")


@app.local_entrypoint()
def main():
    train.remote()