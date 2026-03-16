"""
Modal web endpoint serving ContractReviewAgent with base Mistral 7B.
"""

import modal
import sys

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

agent_mount = modal.Mount.from_local_dir(
    "agent",
    remote_path="/root/agent"
)

from pydantic import BaseModel

class ReviewRequest(BaseModel):
    contract_text: str

class ReviewResponse(BaseModel):
    clauses: dict
    risks: list
    summary: dict


def load_model_and_tokenizer():
    """Load base Mistral 7B — no fine-tuned adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Model ready.")
    return model, tokenizer


@app.cls(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    mounts=[agent_mount],
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=300,
    timeout=600,
)
class ContractReviewService:

    @modal.enter()
    def load(self):
        """Load model once on container startup."""
        sys.path.insert(0, "/root/agent")
        self.model, self.tokenizer = load_model_and_tokenizer()

    @modal.web_endpoint(method="POST")
    def review(self, request: ReviewRequest) -> ReviewResponse:
        from agent import ContractReviewAgent
        agent = ContractReviewAgent(self.model, self.tokenizer)
        result = agent.review_contract(request.contract_text)
        return ReviewResponse(
            clauses=result["clauses"],
            risks=result["risks"],
            summary=result["summary"]
        )