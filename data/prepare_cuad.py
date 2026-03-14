import json
from collections import Counter

def load_cuad(path=r"D:\Aru\NYU\legal-contract-agent\data\CUAD_v1\CUAD_v1\CUAD_v1.json"):
    with open(path) as f:
        raw = json.load(f)
    return raw["data"]

def flatten_examples(data):
    examples = []
    for contract in data:
        title = contract["title"]
        for para in contract["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "contract_title": title,
                    "context": context,
                    "clause_type": qa["question"],
                    "answers": qa["answers"],
                    "is_impossible": qa["is_impossible"]
                })
    return examples

def inspect_sample(examples, n=5):
    print(f"Total examples: {len(examples)}\n")
    for i in range(n):
        ex = examples[i]
        print(f"--- Example {i} ---")
        print(f"Contract    : {ex['contract_title']}")
        print(f"Clause type : {ex['clause_type']}")
        print(f"Context     : {ex['context'][:200]}...")
        print(f"Answer      : {ex['answers']}")
        print(f"Not present : {ex['is_impossible']}")
        print()

def count_by_clause_type(examples):
    counts = Counter(ex["clause_type"] for ex in examples)
    print(f"Total clause types: {len(counts)}\n")
    print("Examples per clause type (top 10):")
    for clause, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}  {clause}")

    # Also show the present/not-present split
    present = sum(1 for ex in examples if not ex["is_impossible"])
    not_present = sum(1 for ex in examples if ex["is_impossible"])
    print(f"\nClause present    : {present}")
    print(f"Clause not present: {not_present}")
    print(f"Ratio             : {not_present/present:.1f}x more negatives than positives")

if __name__ == "__main__":
    data = load_cuad()
    examples = flatten_examples(data)
    inspect_sample(examples)
    count_by_clause_type(examples)