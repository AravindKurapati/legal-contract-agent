import json
from collections import Counter

def load_cuad(path=r"D:\Aru\NYU\legal-contract-agent\data\CUAD_v1\CUAD_v1\CUAD_v1.json"):
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
            # Multiple distinct spans — number them
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


def build_sft_dataset(examples):
    """Format all examples and split into train/val."""
    formatted = [format_for_sft(ex) for ex in examples]
    
    # 90/10 split
    split_idx = int(len(formatted) * 0.9)
    train = formatted[:split_idx]
    val = formatted[split_idx:]
    
    print(f"Train examples: {len(train)}")
    print(f"Val examples  : {len(val)}")
    
    # Save to disk
    with open("data/train_sft.json", "w") as f:
        json.dump(train, f, indent=2)
    
    with open("data/val_sft.json", "w") as f:
        json.dump(val, f, indent=2)
    
    print("\nSaved to data/train_sft.json and data/val_sft.json")
    return train, val

def inspect_formatted(formatted, n=2):
    """Print formatted examples so you can verify they look right."""
    for i in range(n):
        ex = formatted[i]
        print(f"\n--- Formatted Example {i} ---")
        for msg in ex["messages"]:
            print(f"[{msg['role'].upper()}]")
            print(msg["content"][:300])
            print()

if __name__ == "__main__":
    data = load_cuad()
    examples = flatten_examples(data)
    
    print(f"Total examples: {len(examples)}")
    
    train, val = build_sft_dataset(examples)
    inspect_formatted(train, n=2)