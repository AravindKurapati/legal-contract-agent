"""
Modal web endpoint that serves the ContractReviewAgent.

How it works:
- Modal spins up an A10G container on first request (cold start ~15 seconds)
- Model loads once and stays in memory for subsequent requests
- Container idles for 5 minutes then shuts down (no cost while idle)
- Each request runs the full contract review pipeline (~1-2 min per contract)
"""

import modal
import json
import sys
import os

app = modal.App("legal-contract-serve")

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
        "fastapi",
        "pydantic",
    )
)

volume = modal.Volume.from_name("cuad-sft-vol", create_if_missing=True)



from pydantic import BaseModel

class ReviewRequest(BaseModel):
    contract_text: str

class ReviewResponse(BaseModel):
    clauses: dict
    risks: list[str]
    summary: dict



def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path = "/vol/cuad-mistral-final"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("Model ready.")
    return model, tokenizer


# ── Modal function ────────────────────────────────────────────────────────────

@app.cls(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=300,  # shut down after 5 min idle — no cost
    timeout=600,                 # 10 min max per request
)
class ContractReviewService:
    """
    Modal class-based deployment.
    
    Why @app.cls instead of @app.function:
    @app.cls lets us load the model once in @modal.enter()
    and reuse it across multiple requests.
    With @app.function, the model would reload on every request.
    """

    @modal.enter()
    def load(self):
        """Called once when container starts. Loads model into memory."""
        # Add agent directory to path so we can import from it
        sys.path.append("/root/agent")
        self.model, self.tokenizer = load_model_and_tokenizer()

    @modal.web_endpoint(method="POST")
    def review(self, request: ReviewRequest) -> ReviewResponse:

        # Import here because agent files are mounted at runtime
        from agent import ContractReviewAgent

        agent = ContractReviewAgent(self.model, self.tokenizer)
        result = agent.review_contract(request.contract_text)

        return ReviewResponse(
            clauses=result["clauses"],
            risks=result["risks"],
            summary=result["summary"]
        )