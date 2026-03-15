"""
ContractReviewAgent: reviews a full contract across all 41 CUAD clause types.

Pipeline:
1. Load contract text (from PDF or plain text)
2. Chunk into overlapping passages (sliding window)
3. For each clause type, run model on all chunks
4. Collect found clauses, deduplicate
5. Apply risk rules on top of extracted clauses
6. Return structured report

"""

import json
from typing import Optional

CUAD_CLAUSE_TYPES = [
    'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract',
    'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract',
    'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract',
    'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective',
    'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract\'s initial term expire?',
    'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer. Details: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.',
    'Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?',
    'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. Details: Which state/country\'s law governs the interpretation of the contract?',
    'Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be reviewed by a lawyer. Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?',
    'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?',
    'Highlight the parts (if any) of this contract related to "Exclusivity" that should be reviewed by a lawyer. Details: Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all "requirements" from one party of the contract, exclusive rights to use certain technology, products, or services.',
    'Highlight the parts (if any) of this contract related to "No-Solicit Of Customers" that should be reviewed by a lawyer. Details: Is there a restriction on a party\'s soliciting or recruiting customers of the counterparty?',
    'Highlight the parts (if any) of this contract related to "No-Solicit Of Employees" that should be reviewed by a lawyer. Details: Is there a restriction on a party\'s soliciting or recruiting employees of the counterparty?',
    'Highlight the parts (if any) of this contract related to "Non-Disparagement" that should be reviewed by a lawyer. Details: Is there a requirement on a party not to disparage the counterparty?',
    'Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?',
    'Highlight the parts (if any) of this contract related to "ROFR/ROFO/ROFN" that should be reviewed by a lawyer. Details: Is there a right of first refusal, right of first offer or right of first negotiation?',
    'Highlight the parts (if any) of this contract related to "Change Of Control" that should be reviewed by a lawyer. Details: Does one party have the right to terminate or is consent or notice required of the counterparty in case of change of control of the other party?',
    'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer. Details: Is consent or notice required of a party if the contract is assigned to a third party?',
    'Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing" that should be reviewed by a lawyer. Details: Is there a revenue or profit sharing arrangement?',
    'Highlight the parts (if any) of this contract related to "Price Restrictions" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?',
    'Highlight the parts (if any) of this contract related to "Minimum Commitment" that should be reviewed by a lawyer. Details: Is there a minimum order size or minimum amount or units to be purchased?',
    'Highlight the parts (if any) of this contract related to "Volume Restriction" that should be reviewed by a lawyer. Details: Is there a cap on the volume of goods or services?',
    'Highlight the parts (if any) of this contract related to "IP Ownership Assignment" that should be reviewed by a lawyer. Details: Does intellectual property created by one party become the property of the counterparty?',
    'Highlight the parts (if any) of this contract related to "Joint IP Ownership" that should be reviewed by a lawyer. Details: Is there any clause providing for joint or shared ownership of intellectual property between the parties?',
    'Highlight the parts (if any) of this contract related to "License Grant" that should be reviewed by a lawyer. Details: Does the contract contain a license granted by one party to its counterparty?',
    'Highlight the parts (if any) of this contract related to "Non-Transferable License" that should be reviewed by a lawyer. Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?',
    'Highlight the parts (if any) of this contract related to "Affiliate License-Licensor" that should be reviewed by a lawyer. Details: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?',
    'Highlight the parts (if any) of this contract related to "Affiliate License-Licensee" that should be reviewed by a lawyer. Details: Does the contract contain a license grant to a licensee\'s affiliate?',
    'Highlight the parts (if any) of this contract related to "Unlimited/All-You-Can-Eat-License" that should be reviewed by a lawyer. Details: Is there a clause granting one party an "enterprise," "all you can eat" or unlimited usage license?',
    'Highlight the parts (if any) of this contract related to "Irrevocable Or Perpetual License" that should be reviewed by a lawyer. Details: Does the contract contain a perpetual or irrevocable license?',
    'Highlight the parts (if any) of this contract related to "Source Code Escrow" that should be reviewed by a lawyer. Details: Is there a requirement for the contract party to deposit its source code into escrow?',
    'Highlight the parts (if any) of this contract related to "Post-Termination Services" that should be reviewed by a lawyer. Details: Is there a requirement for one party to continue to provide services after the termination of the contract?',
    'Highlight the parts (if any) of this contract related to "Audit Rights" that should be reviewed by a lawyer. Details: Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?',
    'Highlight the parts (if any) of this contract related to "Uncapped Liability" that should be reviewed by a lawyer. Details: Is a party\'s liability uncapped upon the breach of its obligation in the contract?',
    'Highlight the parts (if any) of this contract related to "Cap On Liability" that should be reviewed by a lawyer. Details: Does the contract include a cap on liability upon the breach of a party\'s obligation?',
    'Highlight the parts (if any) of this contract related to "Liquidated Damages" that should be reviewed by a lawyer. Details: Does the contract contain a liquidated damages clause?',
    'Highlight the parts (if any) of this contract related to "Warranty Duration" that should be reviewed by a lawyer. Details: What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?',
    'Highlight the parts (if any) of this contract related to "Insurance" that should be reviewed by a lawyer. Details: Is there a requirement for insurance that must be maintained by one party?',
    'Highlight the parts (if any) of this contract related to "Covenant Not To Sue" that should be reviewed by a lawyer. Details: Is a party restricted from contesting the validity of the counterparty\'s ownership of intellectual property or otherwise bringing a claim against the counterparty?',
    'Highlight the parts (if any) of this contract related to "Third Party Beneficiary" that should be reviewed by a lawyer. Details: Is there a clause providing for a party that is not one of the two primary parties to the contract (e.g., a beneficiary or a guarantor)?',
    'Highlight the parts (if any) of this contract related to "Indemnification" that should be reviewed by a lawyer. Details: Does one party to the contract agree to indemnify the counterparty for losses or expenses?',
]

# Short display names for the UI
CLAUSE_DISPLAY_NAMES = {
    q: q.split('"')[1] for q in CUAD_CLAUSE_TYPES
}


class ContractReviewAgent:
    def __init__(self, model, tokenizer, chunk_size=400, overlap=50):

        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_contract(self, text: str) -> list[str]:

        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

            # Move forward by chunk_size minus overlap
            # This creates the sliding window effect
            if end == len(tokens):
                break
            start += self.chunk_size - self.overlap

        return chunks

    def extract_clause_from_chunk(
        self,
        chunk: str,
        clause_question: str
    ) -> Optional[str]:
        """
        Run model on one chunk for one clause type.
        Returns extracted text, or None if not present.
        """
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
                "content": f"{clause_question}\n\nContract passage:\n{chunk}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        # Return None if model says clause not present
        if "no relevant clause" in response.lower():
            return None

        return response

    def review_contract(self, contract_text: str) -> dict:

        from risk_rules import apply_risk_rules

        print(f"Chunking contract ({len(contract_text)} chars)...")
        chunks = self.chunk_contract(contract_text)
        print(f"Created {len(chunks)} chunks")

        results = {}

        for i, clause_question in enumerate(CUAD_CLAUSE_TYPES):
            display_name = CLAUSE_DISPLAY_NAMES[clause_question]
            found_clauses = []
            seen_texts = set()

            for chunk in chunks:
                result = self.extract_clause_from_chunk(
                    chunk,
                    clause_question
                )

                if result and result not in seen_texts:
                    seen_texts.add(result)
                    found_clauses.append(result)

            results[display_name] = found_clauses

            # Progress update every 10 clauses
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(CUAD_CLAUSE_TYPES)} clause types")

        risk_flags = apply_risk_rules(results)

        present = [k for k, v in results.items() if v]
        missing = [k for k, v in results.items() if not v]

        return {
            "clauses": results,
            "risks": risk_flags,
            "summary": {
                "total_clause_types": len(CUAD_CLAUSE_TYPES),
                "clauses_found": len(present),
                "clauses_missing": len(missing),
                "risk_count": len(risk_flags),
                "present": present,
                "missing": missing,
            }
        }